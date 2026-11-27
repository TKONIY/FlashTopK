# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
#
# The radix selection follows LiteTopK's compact top-k kernel (https://github.com/Heisenberg-Yin/LiteTopK, Apache 2.0).
"""Selection: each row's exact top-k over its candidate buffer, in canonical order, with the call's health, in CuTe DSL.

One CTA per row (256 threads). The candidates are (-score, kv index) pairs, the scan's or the refine's:

  * gated 1-pass fast path (has_gate=True, K < n <= FAST_MAX = 8192): fuses 2-deep 16-byte global loads with a
    scale-invariant 4096-bin (12-bit, packed 16-bit) FP32-log histogram anchored to the row's calibrated
    (origin, inv_delta, th_bucket) span, issues a delayed L2 bulk prefetch of the index slab, finds the pivot bin
    and bucket offsets, emits winners in one pass using per-thread 32-bit bitmasks and one shared atomic per warp,
    batches all index loads before bucket scatter, ranks pivot-bin ties with warp ballots (up to 64 ties), and
    writes canonically ordered outputs directly (skipping score writes when write_scores=False);
  * ungated fast path (DENSE, STAGED, or direct select calls with n <= 8192): fused (count, min, max) load into
    32 KB SMEM + 4096-bin linear histogram + batched gather + direct canonical output write;
  * exact fallback (n > 8192, non-finite/sub-gate score, zero/infinite span, or >256 elements in the pivot bin):
    four 8-bit value radix passes (plus four index passes on a crowded pivot) and log-bucket / sample canonical sort;
  * health: a row whose count passed the buffer (overflow) or that selected fewer than min(k, window) keys
    (underfill) is bad; the first RECOVERY_ROWS bad rows are recorded for recovery.

Health words (int64): 0 overflow rows, 1 underfill rows, 3 the largest candidate count (past the buffer, or always
with stats), 4 the candidate total (with stats), 7 bad rows.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith as _mlir_arith
from cutlass.cutlass_dsl import dsl_user_op

from .common import (
    add_rn,
    ctz_u32,
    div_rn,
    fmax,
    fmin,
    load_global_v4_b32,
    loose_raw_gate,
    mul_rn,
    select_i32,
    store_global_v4_b32,
    sub_rn,
)
from .simt import (
    THREADS,
    WARPS,
    ascending_key,
    atomic_add_global_u64,
    atomic_add_shared_u32,
    atomic_max_global_u64,
    atomic_max_shared_u32,
    atomic_min_shared_u32,
    bit_length_u32,
    bits_f32,
    block_inclusive_scan_i32,
    block_max_i32,
    block_reduce_fin_minmax,
    block_sum_i32,
    cp_async_bulk_prefetch_l2,
    f32_bits,
    f32_to_i32_rz,
    finite,
    load_shared_v4_b32,
    red_add_shared_u32,
    store_shared_v4_b32,
)

RADIX = 256
LOG_BUCKETS = 2048
SORT_BUCKETS = LOG_BUCKETS  # 2048 u32s = 8 KB; also holds the 4096 x 16-bit fast-path histogram
FAST_MAX = 8192  # 32 KB resident sVal buffer in shared memory
HIST12_BINS = 4096
HIST12_WORDS = HIST12_BINS // 2  # 2048 u32s, 2 packed 16-bit counters per word
# Scale-invariant FP32-log binning: 512 bins/exponent across 8 powers of two [ref_span/64, ref_span*4].
_F32LOG_SHIFT = 14
_F32LOG_EXP_BELOW = 6
TIE_CAP = 256  # max pivot-bin elements ranked in registers/SMEM before falling back to 4-pass radix
# A row whose fullest bucket holds more keys falls back to the sample sort (each key's place costs a pass over
# its bucket).
BUCKET_LIMIT = 48
RECOVERY_ROWS = 64
HEALTH_WORDS = 8
H_OVERFLOW, H_UNDERFILL, H_CAND_MAX, H_CAND_SUM, H_BAD = 0, 1, 3, 4, 7
_NEG_INF_BITS = 0xFF800000
_UNROLL = 4
# What a CTA selects from: a row's candidate buffer (the scan's -score, kv index pairs), a slice of a row of dense
# scores (recovery; the index is the column), or the per-slice selections of a row (recovery's second stage).
CANDIDATES, DENSE, STAGED = 0, 1, 2


@cute.jit
def rank_key(v) -> cutlass.Uint32:
    """Ascending float bits with -0.0 tied to +0.0 (matching order_key and canonical_topk)."""
    e = ascending_key(v)
    if (f32_bits(v) & cutlass.Uint32(0x7FFFFFFF)) == cutlass.Uint32(0):
        e = cutlass.Uint32(0x80000000)
    return e


@dsl_user_op
def _u64_max(*, loc=None, ip=None) -> cutlass.Uint64:
    """0xFFFFFFFFFFFFFFFF as a Uint64 IR value without overflowing signed MLIR IntegerAttr(i64)."""
    return cutlass.Uint64(_mlir_arith.constant(cutlass.Uint64.mlir_type, -1, loc=loc, ip=ip))


@cute.jit
def order_key(v, index) -> cutlass.Uint64:
    """Ascending order of this key is canonical order of (-score v, index >= 0): v ascending with -0.0 tying +0.0,
    then the index ascending. The lowest bit keeps a zero's sign, so the key alone gives back v's bits."""
    bits = f32_bits(v)
    e = ascending_key(v)
    negative_zero = cutlass.Uint32(0)
    if (bits & cutlass.Uint32(0x7FFFFFFF)) == cutlass.Uint32(0):
        e = cutlass.Uint32(0x80000000)
        negative_zero = bits >> cutlass.Uint32(31)
    low = (cutlass.Uint32(index) << cutlass.Uint32(1)) | negative_zero
    return (cutlass.Uint64(e) << cutlass.Uint64(32)) | cutlass.Uint64(low)


@cute.jit
def order_index(key) -> Int32:
    return Int32(cutlass.Uint32(key & cutlass.Uint64(0xFFFFFFFF)) >> cutlass.Uint32(1))


@cute.jit
def order_score_bits(key) -> cutlass.Uint32:
    """The score (-v) whose order_key this is."""
    e = cutlass.Uint32(key >> cutlass.Uint64(32))
    v = e ^ cutlass.Uint32(0xFFFFFFFF)
    if (e & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        v = e ^ cutlass.Uint32(0x80000000)
    if (cutlass.Uint32(key & cutlass.Uint64(1))) != cutlass.Uint32(0):
        v = cutlass.Uint32(0x80000000)
    return v ^ cutlass.Uint32(0x80000000)


@cute.jit
def load_value(mVal, row, col, source: cutlass.Constexpr):
    """The entry as -score, the order the selection ranks in."""
    v = mVal[(row, col)]
    if cutlass.const_expr(source != CANDIDATES):
        v = -v
    return v


@cute.jit
def load_index(mIdx, row, col, source: cutlass.Constexpr) -> Int32:
    index = col
    if cutlass.const_expr(source != DENSE):
        index = mIdx[(row, col)]
    return index


@cute.jit
def order_high_score(high) -> Float32:
    """The score whose order_key has this high word (a zero comes back as -0.0)."""
    v = high ^ cutlass.Uint32(0xFFFFFFFF)
    if (high & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        v = high ^ cutlass.Uint32(0x80000000)
    return bits_f32(v ^ cutlass.Uint32(0x80000000))


@cute.jit
def fast_bin12(score, worst, inv_span) -> Int32:
    """Monotone linear 12-bit bucket [0, 4095] of finite score >= worst (higher score -> higher bucket)."""
    b = f32_to_i32_rz(mul_rn(sub_rn(score, worst), inv_span))
    return min(max(b, Int32(0)), Int32(HIST12_BINS - 1))


@cute.jit
def bitonic_sort_asc(sA, tid):
    """Sort sA[0:THREADS] ascending, one element a thread; every thread must reach this."""
    for size_log in cutlass.range_constexpr(1, THREADS.bit_length()):
        size = 1 << size_log
        for j in cutlass.range_constexpr(size_log):
            stride = 1 << (size_log - 1 - j)
            partner = tid ^ stride
            if partner > tid:
                a = sA[tid]
                b = sA[partner]
                if (a > b) == ((tid & size) == 0):
                    sA[tid] = b
                    sA[partner] = a
            cute.arch.sync_threads()


@cute.jit
def radix_bin(sHist, sScan, sState, tid, kfind):
    """The bin holding the kfind-th smallest key (acc < kfind <= acc + hist[bin]): its owner thread writes (bin, acc,
    count) to sState[0..2]. Every thread must reach this; the caller syncs before reading sState."""
    h = sHist[tid]
    incl, _ = block_inclusive_scan_i32(h, sScan, tid)
    acc = incl - h
    if (acc < kfind) & (kfind <= incl):
        sState[0] = tid
        sState[1] = acc
        sState[2] = h


class Select:
    """source CANDIDATES: one CTA per row of the call, with the health. DENSE: `splits` CTAs per slot, each the top-k
    of its slice of the slot's window of dense scores, written to output row map[slot] when splits is 1 and to its
    own row of the stage buffers otherwise. STAGED: one CTA per slot, over the slot's stage rows. A slot whose map
    entry is negative is skipped."""

    def __init__(self, topk: int, source: int = CANDIDATES, write_scores: bool = True, has_gate: bool = False):
        self.topk = topk
        self.source = source
        self.write_scores = write_scores
        self.has_gate = has_gate and (source == CANDIDATES)

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,  # [rows, cap] f32: -score (CANDIDATES) or scores
        idx: cute.Tensor,  # [rows, cap] i32 (unused for DENSE)
        cand_cnt: cute.Tensor,  # [rows] i32: candidates appended, may exceed cap (CANDIDATES)
        ks: cute.Tensor,  # [rows] window starts
        ke: cute.Tensor,
        row_map: cute.Tensor,  # [slots] i32: output row, or -1 (DENSE, STAGED)
        origin: cute.Tensor,  # [rows] f32: seed_prep origin (used when has_gate=True)
        inv_delta: cute.Tensor,  # [rows] f32: seed_prep inv_delta (used when has_gate=True)
        th_bucket: cute.Tensor,  # [rows] i32: seed_prep th_bucket (used when has_gate=True)
        scores: cute.Tensor,  # [out rows, topk] f32 out (unused when write_scores=False)
        indices: cute.Tensor,  # [out rows, topk] int32 out
        health: cute.Tensor,  # [HEALTH_WORDS] i64, accumulated (CANDIDATES)
        bad_rows: cute.Tensor,  # [RECOVERY_ROWS] i32 out (CANDIDATES)
        num_ctas: Int32,
        splits: Int32,
        stats: Int32,
        stream: cuda.CUstream,
    ):
        min_blocks = 4 if self.topk <= 2048 else 2
        self.kernel(
            values, idx, cand_cnt, ks, ke, row_map, origin, inv_delta, th_bucket,
            scores, indices, health, bad_rows, splits, stats,
        ).launch(
            grid=(num_ctas, 1, 1), block=[THREADS, 1, 1], min_blocks_per_mp=min_blocks, stream=stream
        )

    @cute.kernel
    def kernel(
        self, mVal, mIdx, mCnt, mKs, mKe, mMap, mOrigin, mInv, mTh,
        mScores, mIndices, mHealth, mBad, splits: Int32, stats: Int32,
    ):
        K = self.topk
        src = self.source
        items = (K + THREADS - 1) // THREADS
        split_u64 = (K * 3 + 3) // 4 * 2  # in Uint64 units (>= K + K/2, 16-byte aligned)
        arena_u64 = max(FAST_MAX // 2, split_u64 + THREADS)
        place_off = K * 2  # Int32 offset inside sArena (right after sKey[0..K-1])
        ids_u16 = ((max(K, TIE_CAP * 4) + 3) // 4) * 4  # >= K Uint16s and >= TIE_CAP Uint64s (4 KB at K<=2048)
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)

        smem = utils.SmemAllocator()
        # Main reusable SMEM arena (~32 KB at K=2048): sVal[8192 f32] in Phase 1, then sKey/sPlace/sSplit in Phase 2.
        sArena = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((arena_u64,)), 16)
        sVal = cute.recast_tensor(sArena, Float32)
        sKey = sArena
        sPlace = cute.recast_tensor(sArena, Int32)
        sSplit = cute.make_tensor(sArena.iterator + split_u64, cute.make_layout((THREADS,)))
        # 2049 u32s (~8 KB): 4096 x 16-bit histogram / bucket offsets in fast path; sCount & sHist in fallback.
        sCount = smem.allocate_tensor(Int32, cute.make_layout((SORT_BUCKETS + 1,)), 16)
        # 4 KB at K<=2048: holds winning positions (Uint16) in Pass 2, then aliased as sTie (Uint64[256]) in Phase 3.
        sIds = smem.allocate_tensor(cutlass.Uint16, cute.make_layout((ids_u16,)), 16)
        sTie = cute.recast_tensor(sIds, cutlass.Uint64)
        sTiePos = smem.allocate_tensor(cutlass.Uint16, cute.make_layout((TIE_CAP,)), 16)
        sScan = smem.allocate_tensor(Int32, cute.make_layout((WARPS,)), 16)
        sScanF = smem.allocate_tensor(Float32, cute.make_layout((2 * WARPS,)), 16)
        # 0..2 pivot / radix_bin, 3 write counter, 4 / 5 selected keys' smallest / largest high word (or bad flag)
        sState = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
        sVal_addr = sArena.iterator.toint()
        hist = sCount.iterator.toint()
        state = sState.iterator.toint()

        cap = mVal.layout.shape[1]
        row = bid
        out_row = bid
        lo = Int32(0)
        n = cap
        raw_n = Int32(0)
        target = Int32(0)
        active = True
        if cutlass.const_expr(src == CANDIDATES):
            raw_n = mCnt[row]
            n = min(max(raw_n, Int32(0)), cap)
        elif cutlass.const_expr(src == DENSE):
            row = bid // splits
            part = bid - row * splits
            target = mMap[row]
            active = target >= 0
            out_row = select_i32(splits > 1, bid, target)
            start = max(Int32(0), min(cap, mKs[row]))
            end = max(start, min(cap, mKe[row]))
            chunk = (end - start + splits - 1) // splits
            lo = min(end, start + part * chunk)
            n = min(end, lo + chunk) - lo
        else:
            target = mMap[row]
            active = target >= 0
            out_row = target

        if active:
            need_fallback = n > Int32(FAST_MAX)
            done_sorted = cutlass.Boolean(False)
            done_written = cutlass.Boolean(False)
            valid = Int32(0)
            written = Int32(0)
            key_lo = cutlass.Uint32(0xFFFFFFFF)
            key_hi = cutlass.Uint32(0)

            # ----------------------------------------------------------------------------------------------------
            # Path 1 (has_gate=True): 1-pass scale-invariant FP32-log 4096-bin Select + Direct Canonical Write.
            # ----------------------------------------------------------------------------------------------------
            if cutlass.const_expr(self.has_gate):
                if not need_fallback:
                    if n <= Int32(K):
                        # Every finite candidate wins: load directly into sKey, track key_lo/key_hi, generic sort.
                        if tid == 0:
                            sState[3] = Int32(0)
                            sState[4] = Int32(-1)
                            sState[5] = Int32(0)
                        cute.arch.sync_threads()
                        for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                            vs = [mVal[(row, min(base + u * THREADS + tid, n - 1))] for u in range(_UNROLL)]
                            xs = [mIdx[(row, min(base + u * THREADS + tid, n - 1))] for u in range(_UNROLL)]
                            takes = []
                            ballots = []
                            reserve = Int32(0)
                            for u in cutlass.range_constexpr(_UNROLL):
                                take = (base + u * THREADS + tid < n) & finite(vs[u])
                                ballot = cutlass.Uint32(cute.arch.vote_ballot_sync(take))
                                takes.append(take)
                                ballots.append(ballot)
                                reserve = reserve + Int32(cute.arch.popc(ballot))
                            first_s = Int32(0)
                            if (lane == 0) & (reserve > 0):
                                first_s = atomic_add_shared_u32(state + 3 * 4, reserve)
                            first_s = cute.arch.shuffle_sync(first_s, 0)
                            for u in cutlass.range_constexpr(_UNROLL):
                                sl = first_s + Int32(cute.arch.popc(ballots[u] & lt_mask))
                                if takes[u] & (sl < K):
                                    key = order_key(vs[u], xs[u])
                                    sKey[sl] = key
                                    high = cutlass.Uint32(key >> cutlass.Uint64(32))
                                    key_lo = min(key_lo, high)
                                    key_hi = max(key_hi, high)
                                    if xs[u] >= 0:
                                        valid = valid + 1
                                first_s = first_s + Int32(cute.arch.popc(ballots[u]))
                        atomic_min_shared_u32(state + 4 * 4, key_lo)
                        atomic_max_shared_u32(state + 5 * 4, key_hi)
                        valid = block_sum_i32(valid, sScan, tid)
                        written = min(sState[3], Int32(K))
                    else:
                        orig = mOrigin[row]
                        thr = loose_raw_gate(mInv[row], orig, mTh[row])
                        ref_span = sub_rn(-orig, thr)
                        can_f32log = finite(thr) & finite(ref_span) & (ref_span > Float32(0.0))

                        for u in cutlass.range_constexpr(HIST12_WORDS // (THREADS * 4)):
                            store_shared_v4_b32(
                                hist + (tid + u * THREADS) * 16, Int32(0), Int32(0), Int32(0), Int32(0)
                            )
                        if tid == 0:
                            sState[3] = Int32(0)
                            sState[4] = Int32(0)  # non-finite / sub-gate detector
                        cute.arch.sync_threads()

                        row_addr = mVal.iterator.toint() + cutlass.Int64(row) * cutlass.Int64(
                            mVal.layout.stride[0]
                        ) * cutlass.Int64(4)
                        aligned16 = (row_addr & cutlass.Int64(15)) == cutlass.Int64(0)
                        base_bits = cutlass.Uint32(0)

                        if can_f32log:
                            ref_exp = Int32((f32_bits(ref_span) >> cutlass.Uint32(23)) & cutlass.Uint32(0xFF))
                            base_exp = max(ref_exp - Int32(_F32LOG_EXP_BELOW), Int32(1))
                            base_bits = cutlass.Uint32(base_exp) << cutlass.Uint32(23)

                            # Fused Phase 0+1: 2-deep LDG.128 mVal -> sVal + FP32-log 4096-bin histogram!
                            bad = cutlass.Uint32(0)
                            for base in cutlass.range(0, n, THREADS * 8, unroll=1):
                                p0 = base + tid * 4
                                p1 = p0 + THREADS * 4
                                ok0 = aligned16 & (p0 + Int32(4) <= n)
                                ok1 = aligned16 & (p1 + Int32(4) <= n)
                                a0, a1, a2, a3 = Int32(0), Int32(0), Int32(0), Int32(0)
                                b0, b1, b2, b3 = Int32(0), Int32(0), Int32(0), Int32(0)
                                if ok0:
                                    a0, a1, a2, a3 = load_global_v4_b32(row_addr + cutlass.Int64(p0) * cutlass.Int64(4))
                                if ok1:
                                    b0, b1, b2, b3 = load_global_v4_b32(row_addr + cutlass.Int64(p1) * cutlass.Int64(4))
                                if ok0:
                                    store_shared_v4_b32(sVal_addr + p0 * 4, a0, a1, a2, a3)
                                    for aw in (a0, a1, a2, a3):
                                        u = f32_bits(sub_rn(-bits_f32(cutlass.Uint32(aw)), thr))
                                        bad = bad | u | ((u & cutlass.Uint32(0x7F800000)) + cutlass.Uint32(0x00800000))
                                        bi = min(
                                            Int32((max(u, base_bits) - base_bits) >> cutlass.Uint32(_F32LOG_SHIFT)),
                                            Int32(HIST12_BINS - 1),
                                        )
                                        red_add_shared_u32(
                                            hist + (bi >> 1) * 4, select_i32((bi & 1) != 0, Int32(0x10000), Int32(1))
                                        )
                                elif p0 < n:
                                    for u_i in cutlass.range_constexpr(4):
                                        if p0 + u_i < n:
                                            v = mVal[(row, p0 + u_i)]
                                            sVal[p0 + u_i] = v
                                            u = f32_bits(sub_rn(-v, thr))
                                            bad = bad | u | (
                                                (u & cutlass.Uint32(0x7F800000)) + cutlass.Uint32(0x00800000)
                                            )
                                            bi = min(
                                                Int32((max(u, base_bits) - base_bits) >> cutlass.Uint32(_F32LOG_SHIFT)),
                                                Int32(HIST12_BINS - 1),
                                            )
                                            red_add_shared_u32(
                                                hist + (bi >> 1) * 4,
                                                select_i32((bi & 1) != 0, Int32(0x10000), Int32(1)),
                                            )
                                if ok1:
                                    store_shared_v4_b32(sVal_addr + p1 * 4, b0, b1, b2, b3)
                                    for bw in (b0, b1, b2, b3):
                                        u = f32_bits(sub_rn(-bits_f32(cutlass.Uint32(bw)), thr))
                                        bad = bad | u | ((u & cutlass.Uint32(0x7F800000)) + cutlass.Uint32(0x00800000))
                                        bi = min(
                                            Int32((max(u, base_bits) - base_bits) >> cutlass.Uint32(_F32LOG_SHIFT)),
                                            Int32(HIST12_BINS - 1),
                                        )
                                        red_add_shared_u32(
                                            hist + (bi >> 1) * 4, select_i32((bi & 1) != 0, Int32(0x10000), Int32(1))
                                        )
                                elif p1 < n:
                                    for u_i in cutlass.range_constexpr(4):
                                        if p1 + u_i < n:
                                            v = mVal[(row, p1 + u_i)]
                                            sVal[p1 + u_i] = v
                                            u = f32_bits(sub_rn(-v, thr))
                                            bad = bad | u | (
                                                (u & cutlass.Uint32(0x7F800000)) + cutlass.Uint32(0x00800000)
                                            )
                                            bi = min(
                                                Int32((max(u, base_bits) - base_bits) >> cutlass.Uint32(_F32LOG_SHIFT)),
                                                Int32(HIST12_BINS - 1),
                                            )
                                            red_add_shared_u32(
                                                hist + (bi >> 1) * 4,
                                                select_i32((bi & 1) != 0, Int32(0x10000), Int32(1)),
                                            )

                            if (bad & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
                                sState[4] = Int32(1)
                            # Delayed L2 bulk prefetch of mIdx after mVal loads complete so they don't compete for HBM.
                            if tid == 0:
                                idx_row_addr = mIdx.iterator.toint() + cutlass.Int64(row) * cutlass.Int64(
                                    mIdx.layout.stride[0]
                                ) * cutlass.Int64(4)
                                if (idx_row_addr & cutlass.Int64(15)) == cutlass.Int64(0):
                                    pbytes = min(
                                        ((n * Int32(4) + Int32(15)) // Int32(16)) * Int32(16),
                                        (cap * Int32(4)) & Int32(-16),
                                    )
                                    if pbytes > Int32(0):
                                        cp_async_bulk_prefetch_l2(idx_row_addr, pbytes)
                            cute.arch.sync_threads()
                            if sState[4] != Int32(0):
                                need_fallback = cutlass.Boolean(True)
                        else:
                            # Open-gate row (e.g. inv_delta == 0 when window has < k_min samples, K < n <= 8192):
                            # compute exact [worst, best] into sVal and feed the same FP32-log histogram + select!
                            fin = Int32(0)
                            lo_s = Float32(float("inf"))
                            hi_s = Float32(float("-inf"))
                            for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                                pos0 = base + tid * 4
                                if aligned16 & (pos0 + Int32(4) <= n):
                                    w0, w1, w2, w3 = load_global_v4_b32(row_addr + cutlass.Int64(pos0) * cutlass.Int64(4))
                                    store_shared_v4_b32(sVal_addr + pos0 * 4, w0, w1, w2, w3)
                                    for aw in (w0, w1, w2, w3):
                                        v = bits_f32(cutlass.Uint32(aw))
                                        if finite(v):
                                            fin = fin + 1
                                            s = -v
                                            lo_s = fmin(lo_s, s)
                                            hi_s = fmax(hi_s, s)
                                elif pos0 < n:
                                    for u_i in cutlass.range_constexpr(4):
                                        if pos0 + u_i < n:
                                            v = mVal[(row, pos0 + u_i)]
                                            sVal[pos0 + u_i] = v
                                            if finite(v):
                                                fin = fin + 1
                                                s = -v
                                                lo_s = fmin(lo_s, s)
                                                hi_s = fmax(hi_s, s)
                            if tid == 0:
                                idx_row_addr = mIdx.iterator.toint() + cutlass.Int64(row) * cutlass.Int64(
                                    mIdx.layout.stride[0]
                                ) * cutlass.Int64(4)
                                if (idx_row_addr & cutlass.Int64(15)) == cutlass.Int64(0):
                                    pbytes = min(
                                        ((n * Int32(4) + Int32(15)) // Int32(16)) * Int32(16),
                                        (cap * Int32(4)) & Int32(-16),
                                    )
                                    if pbytes > Int32(0):
                                        cp_async_bulk_prefetch_l2(idx_row_addr, pbytes)
                            total_o, worst_o, best_o = block_reduce_fin_minmax(fin, lo_s, hi_s, sScan, sScanF, tid)
                            span_o = sub_rn(best_o, worst_o)
                            if (total_o == n) & finite(span_o) & (span_o > Float32(0.0)):
                                thr = worst_o
                                ref_exp = Int32((f32_bits(span_o) >> cutlass.Uint32(23)) & cutlass.Uint32(0xFF))
                                base_exp = max(ref_exp - Int32(_F32LOG_EXP_BELOW), Int32(1))
                                base_bits = cutlass.Uint32(base_exp) << cutlass.Uint32(23)
                                for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                                    pos0 = base + tid * 4
                                    if pos0 < n:
                                        w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                        ws = (w0, w1, w2, w3)
                                        for u_i in cutlass.range_constexpr(4):
                                            if pos0 + u_i < n:
                                                u = f32_bits(sub_rn(-bits_f32(cutlass.Uint32(ws[u_i])), thr))
                                                bi = min(
                                                    Int32(
                                                        (max(u, base_bits) - base_bits) >> cutlass.Uint32(_F32LOG_SHIFT)
                                                    ),
                                                    Int32(HIST12_BINS - 1),
                                                )
                                                red_add_shared_u32(
                                                    hist + (bi >> 1) * 4,
                                                    select_i32((bi & 1) != 0, Int32(0x10000), Int32(1)),
                                                )
                                cute.arch.sync_threads()
                            else:
                                need_fallback = cutlass.Boolean(True)

                        if not need_fallback:
                            # Descending prefix scan over 4096 packed 16-bit bins + fused max_above.
                            w_base = Int32(HIST12_WORDS - 8) - tid * 8
                            lo0, lo1, lo2, lo3 = load_shared_v4_b32(hist + w_base * 4)
                            hi0, hi1, hi2, hi3 = load_shared_v4_b32(hist + (w_base + 4) * 4)
                            words_desc = (hi3, hi2, hi1, hi0, lo3, lo2, lo1, lo0)
                            counts_hi = []
                            counts_lo = []
                            mine = Int32(0)
                            for k in cutlass.range_constexpr(8):
                                word = words_desc[k]
                                chi = (word >> 16) & 0xFFFF
                                clo = word & 0xFFFF
                                counts_hi.append(chi)
                                counts_lo.append(clo)
                                mine = mine + chi + clo
                            incl, _ = block_inclusive_scan_i32(mine, sScan, tid)
                            running = incl - mine
                            my_max_above = Int32(0)
                            packed_desc = []
                            for k in cutlass.range_constexpr(8):
                                w_idx = w_base + (7 - k)
                                b_hi = w_idx * 2 + 1
                                b_lo = w_idx * 2
                                chi = counts_hi[k]
                                clo = counts_lo[k]
                                off_hi = running
                                if (running < K) & (K <= running + chi):
                                    sState[0] = b_hi
                                    sState[1] = running
                                    sState[2] = chi
                                if running + chi < K:
                                    my_max_above = max(my_max_above, chi)
                                running = running + chi
                                off_lo = running
                                if (running < K) & (K <= running + clo):
                                    sState[0] = b_lo
                                    sState[1] = running
                                    sState[2] = clo
                                if running + clo < K:
                                    my_max_above = max(my_max_above, clo)
                                running = running + clo
                                packed_desc.append(((off_hi & 0xFFFF) << 16) | (off_lo & 0xFFFF))
                            store_shared_v4_b32(
                                hist + w_base * 4, packed_desc[7], packed_desc[6], packed_desc[5], packed_desc[4]
                            )
                            store_shared_v4_b32(
                                hist + (w_base + 4) * 4,
                                packed_desc[3],
                                packed_desc[2],
                                packed_desc[1],
                                packed_desc[0],
                            )
                            wm = my_max_above
                            for off in cutlass.range_constexpr(5):
                                wm = max(wm, cute.arch.shuffle_sync_bfly(wm, 1 << off))
                            if lane == 0:
                                sScan[warp] = wm
                            cute.arch.sync_threads()
                            max_above = sScan[0]
                            for w_i in cutlass.range_constexpr(1, WARPS):
                                max_above = max(max_above, sScan[w_i])
                            pivot_bin = sState[0]
                            count_above = sState[1]
                            pivot_cnt = sState[2]

                            if pivot_cnt > Int32(TIE_CAP):
                                need_fallback = cutlass.Boolean(True)
                            else:
                                # Pass 2: 1 SUB.F32 + 1 unsigned compare per element -> per-thread 32-bit bitmasks!
                                u_hi = cutlass.Uint32(0xFFFFFFFF)
                                if pivot_bin < Int32(HIST12_BINS - 1):
                                    u_hi = base_bits + (cutlass.Uint32(pivot_bin + 1) << cutlass.Uint32(_F32LOG_SHIFT))
                                u_lo = cutlass.Uint32(0)
                                if pivot_bin > Int32(0):
                                    u_lo = base_bits + (cutlass.Uint32(pivot_bin) << cutlass.Uint32(_F32LOG_SHIFT))
                                gt_mask = cutlass.Uint32(0)
                                eq_mask = cutlass.Uint32(0)
                                for g in cutlass.range_constexpr(FAST_MAX // (THREADS * 4)):
                                    pos0 = g * (THREADS * 4) + tid * 4
                                    if pos0 < n:
                                        w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                        ws = (w0, w1, w2, w3)
                                        for u_i in cutlass.range_constexpr(4):
                                            if pos0 + u_i < n:
                                                u = f32_bits(sub_rn(-bits_f32(cutlass.Uint32(ws[u_i])), thr))
                                                bit = cutlass.Uint32(1) << cutlass.Uint32(4 * g + u_i)
                                                if u >= u_hi:
                                                    gt_mask = gt_mask | bit
                                                elif u >= u_lo:
                                                    eq_mask = eq_mask | bit
                                my_gt = Int32(cute.arch.popc(gt_mask))
                                my_eq = Int32(cute.arch.popc(eq_mask))
                                packed_cnt = (my_eq << Int32(16)) | my_gt
                                incl_p = packed_cnt
                                for off in cutlass.range_constexpr(5):
                                    up = cute.arch.shuffle_sync_up(incl_p, 1 << off, mask_and_clamp=0)
                                    if lane >= (1 << off):
                                        incl_p = incl_p + up
                                warp_tot = cute.arch.shuffle_sync(incl_p, 31)
                                base_p = Int32(0)
                                if (lane == 31) & (warp_tot != Int32(0)):
                                    base_p = atomic_add_shared_u32(state + 3 * 4, warp_tot)
                                base_p = cute.arch.shuffle_sync(base_p, 31)
                                my_base = base_p + (incl_p - packed_cnt)
                                gt_slot = my_base & Int32(0xFFFF)
                                eq_slot = (my_base >> Int32(16)) & Int32(0xFFFF)
                                while gt_mask != cutlass.Uint32(0):
                                    e = Int32(ctz_u32(gt_mask))
                                    gt_mask = gt_mask & (gt_mask - cutlass.Uint32(1))
                                    pos = (e >> Int32(2)) * Int32(THREADS * 4) + tid * Int32(4) + (e & Int32(3))
                                    if gt_slot < count_above:
                                        sIds[gt_slot] = cutlass.Uint16(pos)
                                    gt_slot = gt_slot + Int32(1)
                                while eq_mask != cutlass.Uint32(0):
                                    e = Int32(ctz_u32(eq_mask))
                                    eq_mask = eq_mask & (eq_mask - cutlass.Uint32(1))
                                    pos = (e >> Int32(2)) * Int32(THREADS * 4) + tid * Int32(4) + (e & Int32(3))
                                    if eq_slot < Int32(TIE_CAP):
                                        sTiePos[eq_slot] = cutlass.Uint16(pos)
                                    eq_slot = eq_slot + Int32(1)
                                cute.arch.sync_threads()

                                # Phase 3: Batched mIdx gather (all LDGs in flight) before bucket slot reservation.
                                quota = Int32(K) - count_above
                                direct_sort = max_above <= Int32(BUCKET_LIMIT)
                                need_intra = direct_sort & (max_above > Int32(1))
                                vs = []
                                idxs = []
                                for i in cutlass.range_constexpr(items):
                                    slot = tid + i * THREADS
                                    v_i = Float32(0.0)
                                    ix_i = Int32(-1)
                                    if slot < count_above:
                                        p_i = Int32(sIds[slot])
                                        v_i = sVal[p_i]
                                        ix_i = mIdx[(row, p_i)]
                                        if ix_i >= 0:
                                            valid = valid + 1
                                    vs.append(v_i)
                                    idxs.append(ix_i)
                                my_tie = _u64_max()
                                if tid < pivot_cnt:
                                    pos_t = Int32(sTiePos[tid])
                                    my_tie = order_key(sVal[pos_t], mIdx[(row, pos_t)])

                                my_keys = []
                                my_db = []
                                for i in cutlass.range_constexpr(items):
                                    slot = tid + i * THREADS
                                    key_i = cutlass.Uint64(0)
                                    db_i = Int32(K)
                                    if slot < count_above:
                                        key_i = order_key(vs[i], idxs[i])
                                        dst_i = slot
                                        b_i = Int32(0)
                                        if direct_sort:
                                            u = f32_bits(sub_rn(-vs[i], thr))
                                            b_i = min(
                                                Int32(
                                                    (max(u, base_bits) - base_bits) >> cutlass.Uint32(_F32LOG_SHIFT)
                                                ),
                                                Int32(HIST12_BINS - 1),
                                            )
                                            old = atomic_add_shared_u32(
                                                hist + (b_i >> 1) * 4,
                                                select_i32((b_i & 1) != 0, Int32(0x10000), Int32(1)),
                                            )
                                            dst_i = select_i32((b_i & 1) != 0, (old >> 16) & 0xFFFF, old & 0xFFFF)
                                        db_i = (b_i << 13) | dst_i
                                    my_keys.append(key_i)
                                    my_db.append(db_i)

                                # sVal and sIds are now dead: switch arena to sKey/sPlace and alias sIds -> sTie.
                                cute.arch.sync_threads()
                                for i in cutlass.range_constexpr(items):
                                    dst_i = my_db[i] & 0x1FFF
                                    if dst_i < count_above:
                                        sKey[dst_i] = my_keys[i]
                                        if need_intra:
                                            sPlace[place_off + dst_i] = my_db[i] >> 13
                                if tid < pivot_cnt:
                                    sTie[tid] = my_tie
                                cute.arch.sync_threads()

                                # Warp-ballot tie ranking (exact canonical ascending order_key).
                                if pivot_cnt <= Int32(32):
                                    held0 = sTie[lane] if lane < pivot_cnt else _u64_max()
                                    for ti in cutlass.range_constexpr(32 // WARPS):
                                        t = ti * WARPS + warp
                                        if t < pivot_cnt:
                                            tk = sTie[t]
                                            tr = Int32(
                                                cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(held0 < tk)))
                                            )
                                            if (lane == 0) & (tr < quota):
                                                sKey[count_above + tr] = tk
                                                if order_index(tk) >= 0:
                                                    valid = valid + 1
                                elif pivot_cnt <= Int32(64):
                                    held0 = sTie[lane] if lane < pivot_cnt else _u64_max()
                                    held1 = (
                                        sTie[lane + 32]
                                        if lane + 32 < pivot_cnt
                                        else _u64_max()
                                    )
                                    for ti in cutlass.range_constexpr(64 // WARPS):
                                        t = ti * WARPS + warp
                                        if t < pivot_cnt:
                                            tk = sTie[t]
                                            tr = Int32(
                                                cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(held0 < tk)))
                                            ) + Int32(
                                                cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(held1 < tk)))
                                            )
                                            if (lane == 0) & (tr < quota):
                                                sKey[count_above + tr] = tk
                                                if order_index(tk) >= 0:
                                                    valid = valid + 1
                                else:
                                    if tid < pivot_cnt:
                                        tr = Int32(0)
                                        for q in cutlass.range(pivot_cnt, unroll=1):
                                            if sTie[q] < my_tie:
                                                tr = tr + 1
                                        if tr < quota:
                                            sKey[count_above + tr] = my_tie
                                            if order_index(my_tie) >= 0:
                                                valid = valid + 1

                                # Publish tie-ranking writes to sKey AND sum valid across warps in ONE barrier!
                                vw = cute.arch.warp_reduction_sum(valid)
                                if lane == 0:
                                    sScan[warp] = vw
                                cute.arch.sync_threads()
                                if tid == 0:
                                    vt = sScan[0]
                                    for w_i in cutlass.range_constexpr(1, WARPS):
                                        vt = vt + sScan[w_i]
                                    valid = vt
                                written = Int32(K)

                                if direct_sort:
                                    # Half-word sCount lookup + direct canonical global write (-2 CTA barriers!).
                                    for i in cutlass.range_constexpr(items):
                                        slot = tid + i * THREADS
                                        if slot < K:
                                            sk_i = sKey[slot]
                                            place_i = slot
                                            if need_intra & (slot < count_above):
                                                b_i = sPlace[place_off + slot]
                                                w_cur = sCount[b_i >> 1]
                                                last = w_cur & Int32(0xFFFF)
                                                first = (w_cur >> Int32(16)) & Int32(0xFFFF)
                                                if (b_i & Int32(1)) != Int32(0):
                                                    last = first
                                                    first = Int32(0)
                                                    if b_i < Int32(HIST12_BINS - 1):
                                                        first = sCount[(b_i >> 1) + 1] & Int32(0xFFFF)
                                                if last - first > Int32(1):
                                                    place_i = first
                                                    for q in cutlass.range(first, last, unroll=1):
                                                        if sKey[q] < sk_i:
                                                            place_i = place_i + Int32(1)
                                            mIndices[(out_row, place_i)] = order_index(sk_i)
                                            if cutlass.const_expr(self.write_scores):
                                                mScores[(out_row, place_i)] = bits_f32(order_score_bits(sk_i))
                                    done_sorted = cutlass.Boolean(True)
                                    done_written = cutlass.Boolean(True)
                                else:
                                    kl = cutlass.Uint32(0xFFFFFFFF)
                                    kh = cutlass.Uint32(0)
                                    for i in cutlass.range_constexpr(items):
                                        slot = tid + i * THREADS
                                        if slot < K:
                                            h_i = cutlass.Uint32(sKey[slot] >> cutlass.Uint64(32))
                                            kl = min(kl, h_i)
                                            kh = max(kh, h_i)
                                    if tid == 0:
                                        sState[4] = Int32(-1)
                                        sState[5] = Int32(0)
                                    cute.arch.sync_threads()
                                    atomic_min_shared_u32(state + 4 * 4, kl)
                                    atomic_max_shared_u32(state + 5 * 4, kh)
                                    cute.arch.sync_threads()

            # ----------------------------------------------------------------------------------------------------
            # Path 2 (has_gate=False): ungated 2-pass linear 4096-bin SMEM fast path (DENSE, STAGED, direct select).
            # ----------------------------------------------------------------------------------------------------
            else:
                if not need_fallback:
                    for u in cutlass.range_constexpr(HIST12_WORDS // (THREADS * 4)):
                        store_shared_v4_b32(hist + (tid + u * THREADS) * 16, Int32(0), Int32(0), Int32(0), Int32(0))

                    row_addr = mVal.iterator.toint() + (
                        cutlass.Int64(row) * cutlass.Int64(mVal.layout.stride[0]) + cutlass.Int64(lo)
                    ) * cutlass.Int64(4)
                    aligned16 = (row_addr & cutlass.Int64(15)) == cutlass.Int64(0)
                    fin = Int32(0)
                    lo_s = Float32(float("inf"))
                    hi_s = Float32(float("-inf"))
                    for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                        pos0 = base + tid * 4
                        if aligned16 & (pos0 + Int32(4) <= n):
                            w0, w1, w2, w3 = load_global_v4_b32(row_addr + cutlass.Int64(pos0) * cutlass.Int64(4))
                            v0 = bits_f32(cutlass.Uint32(w0))
                            v1 = bits_f32(cutlass.Uint32(w1))
                            v2 = bits_f32(cutlass.Uint32(w2))
                            v3 = bits_f32(cutlass.Uint32(w3))
                            if cutlass.const_expr(src != CANDIDATES):
                                v0, v1, v2, v3 = -v0, -v1, -v2, -v3
                            store_shared_v4_b32(
                                sVal_addr + pos0 * 4,
                                Int32(f32_bits(v0)),
                                Int32(f32_bits(v1)),
                                Int32(f32_bits(v2)),
                                Int32(f32_bits(v3)),
                            )
                            for v in (v0, v1, v2, v3):
                                if finite(v):
                                    fin = fin + 1
                                    s = -v
                                    lo_s = fmin(lo_s, s)
                                    hi_s = fmax(hi_s, s)
                        elif pos0 < n:
                            for u in cutlass.range_constexpr(4):
                                pos = pos0 + u
                                if pos < n:
                                    v = load_value(mVal, row, lo + pos, src)
                                    sVal[pos] = v
                                    if finite(v):
                                        fin = fin + 1
                                        s = -v
                                        lo_s = fmin(lo_s, s)
                                        hi_s = fmax(hi_s, s)
                    if cutlass.const_expr(src != DENSE):
                        if tid == 0:
                            idx_row_addr = mIdx.iterator.toint() + (
                                cutlass.Int64(row) * cutlass.Int64(mIdx.layout.stride[0]) + cutlass.Int64(lo)
                            ) * cutlass.Int64(4)
                            if ((idx_row_addr & cutlass.Int64(15)) == cutlass.Int64(0)) & (n > Int32(0)):
                                pbytes = min(
                                    ((n * Int32(4) + Int32(15)) // Int32(16)) * Int32(16),
                                    ((cap - lo) * Int32(4)) & Int32(-16),
                                )
                                if pbytes > Int32(0):
                                    cp_async_bulk_prefetch_l2(idx_row_addr, pbytes)
                    if tid == 0:
                        sState[3] = Int32(0)
                    total, worst, best = block_reduce_fin_minmax(fin, lo_s, hi_s, sScan, sScanF, tid)

                    if total <= K:
                        take_mask = cutlass.Uint32(0)
                        for g in cutlass.range_constexpr(FAST_MAX // (THREADS * 4)):
                            pos0 = g * (THREADS * 4) + tid * 4
                            if pos0 < n:
                                w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                vs = (
                                    bits_f32(cutlass.Uint32(w0)),
                                    bits_f32(cutlass.Uint32(w1)),
                                    bits_f32(cutlass.Uint32(w2)),
                                    bits_f32(cutlass.Uint32(w3)),
                                )
                                for u in cutlass.range_constexpr(4):
                                    if (pos0 + u < n) & finite(vs[u]):
                                        take_mask = take_mask | (cutlass.Uint32(1) << cutlass.Uint32(4 * g + u))
                        my_cnt = Int32(cute.arch.popc(take_mask))
                        incl_c = my_cnt
                        for off in cutlass.range_constexpr(5):
                            up = cute.arch.shuffle_sync_up(incl_c, 1 << off, mask_and_clamp=0)
                            if lane >= (1 << off):
                                incl_c = incl_c + up
                        warp_tot = cute.arch.shuffle_sync(incl_c, 31)
                        base_c = Int32(0)
                        if (lane == 31) & (warp_tot > Int32(0)):
                            base_c = atomic_add_shared_u32(state + 3 * 4, warp_tot)
                        base_c = cute.arch.shuffle_sync(base_c, 31)
                        out_slot = base_c + (incl_c - my_cnt)
                        while take_mask != cutlass.Uint32(0):
                            e = Int32(ctz_u32(take_mask))
                            take_mask = take_mask & (take_mask - cutlass.Uint32(1))
                            pos = (e >> Int32(2)) * Int32(THREADS * 4) + tid * Int32(4) + (e & Int32(3))
                            if out_slot < Int32(K):
                                sIds[out_slot] = cutlass.Uint16(pos)
                            out_slot = out_slot + Int32(1)
                        cute.arch.sync_threads()
                        written = min(sState[3], Int32(K))
                        my_keys = []
                        for i in cutlass.range_constexpr(items):
                            slot = tid + i * THREADS
                            key_i = cutlass.Uint64(0)
                            if slot < written:
                                pos_i = Int32(sIds[slot])
                                v_i = sVal[pos_i]
                                idx_i = load_index(mIdx, row, lo + pos_i, src)
                                key_i = order_key(v_i, idx_i)
                                high = cutlass.Uint32(key_i >> cutlass.Uint64(32))
                                key_lo = min(key_lo, high)
                                key_hi = max(key_hi, high)
                                if idx_i >= 0:
                                    valid = valid + 1
                            my_keys.append(key_i)
                        if tid == 0:
                            sState[4] = Int32(-1)
                            sState[5] = Int32(0)
                        cute.arch.sync_threads()
                        for i in cutlass.range_constexpr(items):
                            slot = tid + i * THREADS
                            if slot < written:
                                sKey[slot] = my_keys[i]
                        atomic_min_shared_u32(state + 4 * 4, key_lo)
                        atomic_max_shared_u32(state + 5 * 4, key_hi)
                        valid = block_sum_i32(valid, sScan, tid)
                    else:
                        span = sub_rn(best, worst)
                        inv_span = div_rn(Float32(HIST12_BINS - 1), span)
                        can_hist = finite(span) & (span > Float32(0.0)) & finite(inv_span) & (inv_span > Float32(0.0))
                        if not can_hist:
                            need_fallback = cutlass.Boolean(True)
                        else:
                            for base in cutlass.range(0, n, THREADS * 4, unroll=1):
                                pos0 = base + tid * 4
                                if pos0 < n:
                                    w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                    vs = (
                                        bits_f32(cutlass.Uint32(w0)),
                                        bits_f32(cutlass.Uint32(w1)),
                                        bits_f32(cutlass.Uint32(w2)),
                                        bits_f32(cutlass.Uint32(w3)),
                                    )
                                    for u in cutlass.range_constexpr(4):
                                        if (pos0 + u < n) & finite(vs[u]):
                                            b = fast_bin12(-vs[u], worst, inv_span)
                                            red_add_shared_u32(
                                                hist + (b >> 1) * 4,
                                                select_i32((b & 1) != 0, Int32(0x10000), Int32(1)),
                                            )
                            cute.arch.sync_threads()

                            w_base = Int32(HIST12_WORDS - 8) - tid * 8
                            lo0, lo1, lo2, lo3 = load_shared_v4_b32(hist + w_base * 4)
                            hi0, hi1, hi2, hi3 = load_shared_v4_b32(hist + (w_base + 4) * 4)
                            words_desc = (hi3, hi2, hi1, hi0, lo3, lo2, lo1, lo0)
                            counts_hi = []
                            counts_lo = []
                            mine = Int32(0)
                            for k in cutlass.range_constexpr(8):
                                word = words_desc[k]
                                chi = (word >> 16) & 0xFFFF
                                clo = word & 0xFFFF
                                counts_hi.append(chi)
                                counts_lo.append(clo)
                                mine = mine + chi + clo
                            incl, _ = block_inclusive_scan_i32(mine, sScan, tid)
                            running = incl - mine
                            my_max_above = Int32(0)
                            packed_desc = []
                            for k in cutlass.range_constexpr(8):
                                w_idx = w_base + (7 - k)
                                b_hi = w_idx * 2 + 1
                                b_lo = w_idx * 2
                                chi = counts_hi[k]
                                clo = counts_lo[k]
                                off_hi = running
                                if (running < K) & (K <= running + chi):
                                    sState[0] = b_hi
                                    sState[1] = running
                                    sState[2] = chi
                                if running + chi < K:
                                    my_max_above = max(my_max_above, chi)
                                running = running + chi
                                off_lo = running
                                if (running < K) & (K <= running + clo):
                                    sState[0] = b_lo
                                    sState[1] = running
                                    sState[2] = clo
                                if running + clo < K:
                                    my_max_above = max(my_max_above, clo)
                                running = running + clo
                                packed_desc.append(((off_hi & 0xFFFF) << 16) | (off_lo & 0xFFFF))
                            store_shared_v4_b32(
                                hist + w_base * 4, packed_desc[7], packed_desc[6], packed_desc[5], packed_desc[4]
                            )
                            store_shared_v4_b32(
                                hist + (w_base + 4) * 4,
                                packed_desc[3],
                                packed_desc[2],
                                packed_desc[1],
                                packed_desc[0],
                            )
                            wm = my_max_above
                            for off in cutlass.range_constexpr(5):
                                wm = max(wm, cute.arch.shuffle_sync_bfly(wm, 1 << off))
                            if lane == 0:
                                sScan[warp] = wm
                            cute.arch.sync_threads()
                            max_above = sScan[0]
                            for w_i in cutlass.range_constexpr(1, WARPS):
                                max_above = max(max_above, sScan[w_i])
                            pivot_bin = sState[0]
                            count_above = sState[1]
                            pivot_cnt = sState[2]

                            if pivot_cnt > Int32(TIE_CAP):
                                need_fallback = cutlass.Boolean(True)
                            else:
                                gt_mask = cutlass.Uint32(0)
                                eq_mask = cutlass.Uint32(0)
                                for g in cutlass.range_constexpr(FAST_MAX // (THREADS * 4)):
                                    pos0 = g * (THREADS * 4) + tid * 4
                                    if pos0 < n:
                                        w0, w1, w2, w3 = load_shared_v4_b32(sVal_addr + pos0 * 4)
                                        vs = (
                                            bits_f32(cutlass.Uint32(w0)),
                                            bits_f32(cutlass.Uint32(w1)),
                                            bits_f32(cutlass.Uint32(w2)),
                                            bits_f32(cutlass.Uint32(w3)),
                                        )
                                        for u in cutlass.range_constexpr(4):
                                            pos = pos0 + u
                                            if (pos < n) & finite(vs[u]):
                                                bu = fast_bin12(-vs[u], worst, inv_span)
                                                bit = cutlass.Uint32(1) << cutlass.Uint32(4 * g + u)
                                                if bu > pivot_bin:
                                                    gt_mask = gt_mask | bit
                                                elif bu == pivot_bin:
                                                    eq_mask = eq_mask | bit
                                my_gt = Int32(cute.arch.popc(gt_mask))
                                my_eq = Int32(cute.arch.popc(eq_mask))
                                packed_cnt = (my_eq << Int32(16)) | my_gt
                                incl_p = packed_cnt
                                for off in cutlass.range_constexpr(5):
                                    up = cute.arch.shuffle_sync_up(incl_p, 1 << off, mask_and_clamp=0)
                                    if lane >= (1 << off):
                                        incl_p = incl_p + up
                                warp_tot = cute.arch.shuffle_sync(incl_p, 31)
                                base_p = Int32(0)
                                if (lane == 31) & (warp_tot != Int32(0)):
                                    base_p = atomic_add_shared_u32(state + 3 * 4, warp_tot)
                                base_p = cute.arch.shuffle_sync(base_p, 31)
                                my_base = base_p + (incl_p - packed_cnt)
                                gt_slot = my_base & Int32(0xFFFF)
                                eq_slot = (my_base >> Int32(16)) & Int32(0xFFFF)
                                while gt_mask != cutlass.Uint32(0):
                                    e = Int32(ctz_u32(gt_mask))
                                    gt_mask = gt_mask & (gt_mask - cutlass.Uint32(1))
                                    pos = (e >> Int32(2)) * Int32(THREADS * 4) + tid * Int32(4) + (e & Int32(3))
                                    if gt_slot < count_above:
                                        sIds[gt_slot] = cutlass.Uint16(pos)
                                    gt_slot = gt_slot + Int32(1)
                                while eq_mask != cutlass.Uint32(0):
                                    e = Int32(ctz_u32(eq_mask))
                                    eq_mask = eq_mask & (eq_mask - cutlass.Uint32(1))
                                    pos = (e >> Int32(2)) * Int32(THREADS * 4) + tid * Int32(4) + (e & Int32(3))
                                    if eq_slot < Int32(TIE_CAP):
                                        sTiePos[eq_slot] = cutlass.Uint16(pos)
                                    eq_slot = eq_slot + Int32(1)
                                cute.arch.sync_threads()

                                quota = Int32(K) - count_above
                                direct_sort = max_above <= Int32(BUCKET_LIMIT)
                                need_intra = direct_sort & (max_above > Int32(1))
                                vs = []
                                idxs = []
                                for i in cutlass.range_constexpr(items):
                                    slot = tid + i * THREADS
                                    v_i = Float32(0.0)
                                    ix_i = Int32(-1)
                                    if slot < count_above:
                                        p_i = Int32(sIds[slot])
                                        v_i = sVal[p_i]
                                        ix_i = load_index(mIdx, row, lo + p_i, src)
                                        if ix_i >= 0:
                                            valid = valid + 1
                                    vs.append(v_i)
                                    idxs.append(ix_i)
                                my_tie = _u64_max()
                                if tid < pivot_cnt:
                                    pos_t = Int32(sTiePos[tid])
                                    my_tie = order_key(sVal[pos_t], load_index(mIdx, row, lo + pos_t, src))

                                my_keys = []
                                my_db = []
                                for i in cutlass.range_constexpr(items):
                                    slot = tid + i * THREADS
                                    key_i = cutlass.Uint64(0)
                                    db_i = Int32(K)
                                    if slot < count_above:
                                        key_i = order_key(vs[i], idxs[i])
                                        dst_i = slot
                                        b_i = Int32(0)
                                        if direct_sort:
                                            b_i = fast_bin12(-vs[i], worst, inv_span)
                                            old = atomic_add_shared_u32(
                                                hist + (b_i >> 1) * 4,
                                                select_i32((b_i & 1) != 0, Int32(0x10000), Int32(1)),
                                            )
                                            dst_i = select_i32((b_i & 1) != 0, (old >> 16) & 0xFFFF, old & 0xFFFF)
                                        db_i = (b_i << 13) | dst_i
                                    my_keys.append(key_i)
                                    my_db.append(db_i)

                                cute.arch.sync_threads()
                                for i in cutlass.range_constexpr(items):
                                    dst_i = my_db[i] & 0x1FFF
                                    if dst_i < count_above:
                                        sKey[dst_i] = my_keys[i]
                                        if need_intra:
                                            sPlace[place_off + dst_i] = my_db[i] >> 13
                                if tid < pivot_cnt:
                                    sTie[tid] = my_tie
                                cute.arch.sync_threads()

                                if pivot_cnt <= Int32(32):
                                    held0 = sTie[lane] if lane < pivot_cnt else _u64_max()
                                    for ti in cutlass.range_constexpr(32 // WARPS):
                                        t = ti * WARPS + warp
                                        if t < pivot_cnt:
                                            tk = sTie[t]
                                            tr = Int32(
                                                cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(held0 < tk)))
                                            )
                                            if (lane == 0) & (tr < quota):
                                                sKey[count_above + tr] = tk
                                                if order_index(tk) >= 0:
                                                    valid = valid + 1
                                elif pivot_cnt <= Int32(64):
                                    held0 = sTie[lane] if lane < pivot_cnt else _u64_max()
                                    held1 = (
                                        sTie[lane + 32]
                                        if lane + 32 < pivot_cnt
                                        else _u64_max()
                                    )
                                    for ti in cutlass.range_constexpr(64 // WARPS):
                                        t = ti * WARPS + warp
                                        if t < pivot_cnt:
                                            tk = sTie[t]
                                            tr = Int32(
                                                cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(held0 < tk)))
                                            ) + Int32(
                                                cute.arch.popc(cutlass.Uint32(cute.arch.vote_ballot_sync(held1 < tk)))
                                            )
                                            if (lane == 0) & (tr < quota):
                                                sKey[count_above + tr] = tk
                                                if order_index(tk) >= 0:
                                                    valid = valid + 1
                                else:
                                    if tid < pivot_cnt:
                                        tr = Int32(0)
                                        for q in cutlass.range(pivot_cnt, unroll=1):
                                            if sTie[q] < my_tie:
                                                tr = tr + 1
                                        if tr < quota:
                                            sKey[count_above + tr] = my_tie
                                            if order_index(my_tie) >= 0:
                                                valid = valid + 1

                                valid = block_sum_i32(valid, sScan, tid)
                                written = Int32(K)

                                if direct_sort:
                                    for i in cutlass.range_constexpr(items):
                                        slot = tid + i * THREADS
                                        if slot < K:
                                            sk_i = sKey[slot]
                                            place_i = slot
                                            if need_intra & (slot < count_above):
                                                b_i = sPlace[place_off + slot]
                                                w_cur = sCount[b_i >> 1]
                                                last = w_cur & Int32(0xFFFF)
                                                first = (w_cur >> Int32(16)) & Int32(0xFFFF)
                                                if (b_i & Int32(1)) != Int32(0):
                                                    last = first
                                                    first = Int32(0)
                                                    if b_i < Int32(HIST12_BINS - 1):
                                                        first = sCount[(b_i >> 1) + 1] & Int32(0xFFFF)
                                                if last - first > Int32(1):
                                                    place_i = first
                                                    for q in cutlass.range(first, last, unroll=1):
                                                        if sKey[q] < sk_i:
                                                            place_i = place_i + Int32(1)
                                            mIndices[(out_row, place_i)] = order_index(sk_i)
                                            if cutlass.const_expr(self.write_scores):
                                                mScores[(out_row, place_i)] = bits_f32(order_score_bits(sk_i))
                                    done_sorted = cutlass.Boolean(True)
                                    done_written = cutlass.Boolean(True)
                                else:
                                    kl = cutlass.Uint32(0xFFFFFFFF)
                                    kh = cutlass.Uint32(0)
                                    for i in cutlass.range_constexpr(items):
                                        slot = tid + i * THREADS
                                        if slot < K:
                                            h_i = cutlass.Uint32(sKey[slot] >> cutlass.Uint64(32))
                                            kl = min(kl, h_i)
                                            kh = max(kh, h_i)
                                    if tid == 0:
                                        sState[4] = Int32(-1)
                                        sState[5] = Int32(0)
                                    cute.arch.sync_threads()
                                    atomic_min_shared_u32(state + 4 * 4, kl)
                                    atomic_max_shared_u32(state + 5 * 4, kh)
                                    cute.arch.sync_threads()

            # ----------------------------------------------------------------------------------------------------
            # Exact 4-pass radix fallback (n > FAST_MAX, non-finite/sub-gate, zero/inf span, or >256 pivot ties).
            # ----------------------------------------------------------------------------------------------------
            if need_fallback:
                sCount[tid] = Int32(0)
                cute.arch.sync_threads()
                fin = Int32(0)
                for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                    vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                    for u in cutlass.range_constexpr(_UNROLL):
                        if (base + u * THREADS + tid < n) & finite(vs[u]):
                            fin = fin + 1
                            red_add_shared_u32(hist + Int32(rank_key(vs[u]) >> cutlass.Uint32(24)) * 4, Int32(1))
                total = block_sum_i32(fin, sScan, tid)
                target = min(total, Int32(K))

                pivot = cutlass.Uint32(0xFFFFFFFF)
                tie_cut = cutlass.Uint32(0xFFFFFFFF)
                if total > K:
                    desired = cutlass.Uint32(0)
                    mask = cutlass.Uint32(0)
                    kfind = target
                    below = Int32(0)
                    equal = Int32(0)
                    for p in cutlass.range_constexpr(4):
                        shift = 24 - 8 * p
                        if p > 0:
                            sCount[tid] = Int32(0)
                            cute.arch.sync_threads()
                            for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                                vs = [
                                    load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src)
                                    for u in range(_UNROLL)
                                ]
                                for u in cutlass.range_constexpr(_UNROLL):
                                    e = rank_key(vs[u])
                                    if (base + u * THREADS + tid < n) & finite(vs[u]) & ((e & mask) == desired):
                                        b = Int32((e >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                                        red_add_shared_u32(hist + b * 4, Int32(1))
                            cute.arch.sync_threads()
                        radix_bin(sCount, sScan, sState, tid, kfind)
                        cute.arch.sync_threads()
                        desired = desired | (cutlass.Uint32(sState[0]) << cutlass.Uint32(shift))
                        kfind = kfind - sState[1]
                        below = below + sState[1]
                        equal = sState[2]
                        mask = mask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                        cute.arch.sync_threads()
                    pivot = desired
                    eq_take = target - below
                    if equal > eq_take:
                        idesired = cutlass.Uint32(0)
                        imask = cutlass.Uint32(0)
                        ikfind = eq_take
                        for p in cutlass.range_constexpr(4):
                            shift = 24 - 8 * p
                            sCount[tid] = Int32(0)
                            cute.arch.sync_threads()
                            for j in cutlass.range(tid, n, THREADS, unroll=1):
                                v = load_value(mVal, row, lo + j, src)
                                if finite(v):
                                    if rank_key(v) == pivot:
                                        ik = cutlass.Uint32(load_index(mIdx, row, lo + j, src))
                                        if (ik & imask) == idesired:
                                            b = Int32((ik >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                                            red_add_shared_u32(hist + b * 4, Int32(1))
                            cute.arch.sync_threads()
                            radix_bin(sCount, sScan, sState, tid, ikfind)
                            cute.arch.sync_threads()
                            idesired = idesired | (cutlass.Uint32(sState[0]) << cutlass.Uint32(shift))
                            ikfind = ikfind - sState[1]
                            imask = imask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                            cute.arch.sync_threads()
                        tie_cut = idesired

                if tid == 0:
                    sState[3] = Int32(0)
                    sState[4] = Int32(-1)
                    sState[5] = Int32(0)
                cute.arch.sync_threads()
                key_lo = cutlass.Uint32(0xFFFFFFFF)
                key_hi = cutlass.Uint32(0)
                valid = Int32(0)
                for base in cutlass.range(0, n, THREADS * _UNROLL, unroll=1):
                    vs = [load_value(mVal, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                    xs = [load_index(mIdx, row, lo + min(base + u * THREADS + tid, n - 1), src) for u in range(_UNROLL)]
                    takes = []
                    ballots = []
                    reserve = Int32(0)
                    for u in cutlass.range_constexpr(_UNROLL):
                        e = rank_key(vs[u])
                        take = (
                            (base + u * THREADS + tid < n)
                            & finite(vs[u])
                            & ((e < pivot) | ((e == pivot) & (cutlass.Uint32(xs[u]) <= tie_cut)))
                        )
                        ballot = cutlass.Uint32(cute.arch.vote_ballot_sync(take))
                        takes.append(take)
                        ballots.append(ballot)
                        reserve = reserve + Int32(cute.arch.popc(ballot))
                    first = Int32(0)
                    if (lane == 0) & (reserve > 0):
                        first = atomic_add_shared_u32(state + 3 * 4, reserve)
                    first = cute.arch.shuffle_sync(first, 0)
                    for u in cutlass.range_constexpr(_UNROLL):
                        slot = first + Int32(cute.arch.popc(ballots[u] & lt_mask))
                        if takes[u] & (slot < K):
                            key = order_key(vs[u], xs[u])
                            sKey[slot] = key
                            high = cutlass.Uint32(key >> cutlass.Uint64(32))
                            key_lo = min(key_lo, high)
                            key_hi = max(key_hi, high)
                            if xs[u] >= 0:
                                valid = valid + 1
                        first = first + Int32(cute.arch.popc(ballots[u]))
                atomic_min_shared_u32(state + 4 * 4, key_lo)
                atomic_max_shared_u32(state + 5 * 4, key_hi)
                valid = block_sum_i32(valid, sScan, tid)
                written = min(sState[3], Int32(K))

            # ----------------------------------------------------------------------------------------------------
            # Generic canonical sort (for n <= K, fallback, or max_above > BUCKET_LIMIT) + global writeback.
            # ----------------------------------------------------------------------------------------------------
            if not done_sorted:
                for b in cutlass.range(tid, SORT_BUCKETS + 1, THREADS, unroll=1):
                    sCount[b] = Int32(0)
                cute.arch.sync_threads()
                best = order_high_score(cutlass.Uint32(sState[4]))
                worst = order_high_score(cutlass.Uint32(sState[5]))
                span = sub_rn(best, worst)
                eps = mul_rn(span, Float32(2.0**-13))
                base = f32_bits(eps)
                shift = max(bit_length_u32(f32_bits(add_rn(span, eps)) - base) - 11, Int32(0))
                spread = finite(span) & (span > Float32(0.0))
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    if slot < written:
                        s_key = order_high_score(cutlass.Uint32(sKey[slot] >> cutlass.Uint64(32)))
                        up = Int32((f32_bits(add_rn(sub_rn(s_key, worst), eps)) - base) >> cutlass.Uint32(shift))
                        d = Int32(LOG_BUCKETS - 1) - min(max(up, Int32(0)), Int32(LOG_BUCKETS - 1))
                        d = select_i32(spread, d, Int32(0))
                        rank = atomic_add_shared_u32(sCount.iterator.toint() + d * 4, Int32(1))
                        sPlace[place_off + slot] = (d << 16) | rank
                cute.arch.sync_threads()
                fullest = Int32(0)
                for u in cutlass.range_constexpr(LOG_BUCKETS // THREADS):
                    fullest = max(fullest, sCount[tid * (LOG_BUCKETS // THREADS) + u])
                fullest = block_max_i32(fullest, sScan, tid)
                if fullest > BUCKET_LIMIT:
                    for b in cutlass.range(tid, SORT_BUCKETS + 1, THREADS, unroll=1):
                        sCount[b] = Int32(0)
                    splitter = cutlass.Uint64(0)
                    if written > 0:
                        splitter = sKey[min((tid * written) // THREADS, Int32(K - 1))]
                    sSplit[tid] = splitter
                    cute.arch.sync_threads()
                    bitonic_sort_asc(sSplit, tid)
                    for i in cutlass.range_constexpr(items):
                        slot = tid + i * THREADS
                        if slot < written:
                            key = sKey[slot]
                            pos = Int32(0)
                            for j in cutlass.range_constexpr(THREADS.bit_length() - 1):
                                half = THREADS >> (j + 1)
                                if sSplit[pos + half - 1] < key:
                                    pos = pos + half
                            if sSplit[pos] < key:
                                pos = pos + 1
                            rank = atomic_add_shared_u32(sCount.iterator.toint() + pos * 4, Int32(1))
                            sPlace[place_off + slot] = (pos << 16) | rank
                    cute.arch.sync_threads()
                per = SORT_BUCKETS // THREADS
                counts = [sCount[tid * per + u] for u in range(per)]
                mine = Int32(0)
                for u in cutlass.range_constexpr(per):
                    mine = mine + counts[u]
                incl, _ = block_inclusive_scan_i32(mine, sScan, tid)
                run = incl - mine
                for u in cutlass.range_constexpr(per):
                    sCount[tid * per + u] = run
                    run = run + counts[u]
                if tid == THREADS - 1:
                    sCount[SORT_BUCKETS] = incl
                cute.arch.sync_threads()

                keys = []
                packed_places = []
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    at = min(slot, Int32(K - 1))
                    keys.append(sKey[at])
                    packed_places.append(select_i32(slot < written, sPlace[place_off + at], Int32(SORT_BUCKETS << 16)))
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(items):
                    d = packed_places[i] >> 16
                    if d < SORT_BUCKETS:
                        dst = sCount[d] + (packed_places[i] & 0xFFFF)
                        sKey[dst] = keys[i]
                        sPlace[place_off + dst] = d
                cute.arch.sync_threads()

                sorted_keys = []
                sorted_places = []
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    at = min(slot, Int32(K - 1))
                    key_fb = sKey[at]
                    place_i = Int32(K)
                    if slot < written:
                        d = sPlace[place_off + at]
                        first = sCount[d]
                        last = sCount[d + 1]
                        place_i = first
                        if last - first > 1:
                            for q in cutlass.range(first, last, unroll=1):
                                if sKey[q] < key_fb:
                                    place_i = place_i + 1
                    sorted_keys.append(key_fb)
                    sorted_places.append(place_i)
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(items):
                    if sorted_places[i] < K:
                        sKey[sorted_places[i]] = sorted_keys[i]
                cute.arch.sync_threads()

            if not done_written:
                for i in cutlass.range_constexpr(items):
                    slot = tid + i * THREADS
                    if slot < K:
                        if slot < written:
                            key_out = sKey[slot]
                            if cutlass.const_expr(self.write_scores):
                                mScores[(out_row, slot)] = bits_f32(order_score_bits(key_out))
                            mIndices[(out_row, slot)] = order_index(key_out)
                        else:
                            if cutlass.const_expr(self.write_scores):
                                mScores[(out_row, slot)] = bits_f32(cutlass.Uint32(_NEG_INF_BITS))
                            mIndices[(out_row, slot)] = Int32(-1)

            if cutlass.const_expr(src == CANDIDATES):
                if tid == 0:
                    expected = min(max(mKe[row] - mKs[row], Int32(0)), Int32(K))
                    health = mHealth.iterator.toint()
                    overflow = raw_n > cap
                    underfill = valid < expected
                    if overflow:
                        atomic_add_global_u64(health + H_OVERFLOW * 8, cutlass.Int64(1))
                    if underfill:
                        atomic_add_global_u64(health + H_UNDERFILL * 8, cutlass.Int64(1))
                    if overflow | underfill:
                        bad_slot = atomic_add_global_u64(health + H_BAD * 8, cutlass.Int64(1))
                        if bad_slot < RECOVERY_ROWS:
                            mBad[Int32(bad_slot)] = row
                    candidates = cutlass.Int64(max(raw_n, Int32(0)))
                    if (stats != 0) | overflow:
                        atomic_max_global_u64(health + H_CAND_MAX * 8, candidates)
                    if stats != 0:
                        atomic_add_global_u64(health + H_CAND_SUM * 8, candidates)
