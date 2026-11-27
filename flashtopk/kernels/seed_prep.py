# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
#
# The calibration follows LiteTopK's seed-prep kernel (https://github.com/Heisenberg-Yin/LiteTopK, Apache 2.0).
"""Gate calibration: each row's bucket map and seed gate from its shared-sample scores, in CuTe DSL.

One CTA per row (256 threads). The row's samples are the columns [col_lo, col_hi) of its sample-score row:

  * pass 1: the finite samples' minimum, maximum and count;
  * the bucket map: origin o = -max, span = max(max(-min - o, |.| 2^-8), 1e-6), inv_delta = (NB - 1) / span, so a
    score s falls in bucket trunc(fma(-inv, s, -o * inv)), clamped to [0, NB);
  * pass 2: the samples' bucket histogram;
  * the gate: the bucket of the k-th best sample. k is topk when the samples are the row's whole window; otherwise
    the samples are spread over the window, so the k-th best sits near the (k * window / samples)-th best key, and
    k = ceil(slack_topk * samples / window), clamped to [k_min, topk], aims the gate at about slack_topk candidates;
  * a row with fewer finite samples than k gets inv_delta = 0: every key lands in bucket 0, the gate admits the
    window.

Every float operation is written with its rounding (no contraction, IEEE division), so the outputs are the bits the
scan's gate expects whatever the compiler does.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32

from .common import abs_f32, add_rn, div_rn, fma_rn, fmax, fmin, i32_to_f32_rn, mul_rn, sqrt_rn, sub_rn
from .simt import THREADS, WARPS, atomic_min_shared_s32, red_add_shared_u32, block_inclusive_scan_i32, ceil_f32, \
    f32_to_i32_rz, finite

NUM_BUCKETS = 256
NSUB = 4  # sub-histograms, to spread shared-memory atomics


class SeedPrep:
    @cute.jit
    def __call__(
        self,
        slog: cute.Tensor,  # [num_q, width] f32 sample scores
        ks: cute.Tensor,  # [num_q] i32
        ke: cute.Tensor,  # [num_q] i32
        col_lo: cute.Tensor,  # [num_q] i32: the row's first sample column
        col_hi: cute.Tensor,  # [num_q] i32: one past its last
        origin: cute.Tensor,  # [num_q] f32 out
        inv_delta: cute.Tensor,  # [num_q] f32 out
        th_bucket: cute.Tensor,  # [num_q] i32 out (pruning gate)
        th_bucket_scan: cute.Tensor,  # [num_q] i32 out (4-sigma scan gate; may alias th_bucket)
        cand_cnt: cute.Tensor,  # [num_q] i32 out: zeroed
        num_q: Int32,
        topk: Int32,
        slack_topk: Float32,
        k_min: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(slog, ks, ke, col_lo, col_hi, origin, inv_delta, th_bucket, th_bucket_scan, cand_cnt, topk,
                    slack_topk, k_min).launch(grid=(num_q, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mS, mKs, mKe, mLo, mHi, mOrigin, mInv, mTh, mThScan, mCnt, topk: Int32, slack_topk: Float32,
               k_min: Int32):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tid // 32
        smem = utils.SmemAllocator()
        sHist = smem.allocate_tensor(Int32, cute.make_layout((NSUB * NUM_BUCKETS,)), 16)
        sMax = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        sMin = smem.allocate_tensor(Float32, cute.make_layout((WARPS,)), 16)
        sCount = smem.allocate_tensor(Int32, cute.make_layout((WARPS,)), 16)
        sScan = smem.allocate_tensor(Int32, cute.make_layout((WARPS,)), 16)
        sTh = smem.allocate_tensor(Int32, cute.make_layout((2,)), 16)

        width = mS.layout.shape[1]
        c0 = max(mLo[row], Int32(0))
        c1 = min(mHi[row], width)

        # pass 1: min / max / count of the finite samples
        mx = Float32(float("-inf"))
        mn = Float32(float("inf"))
        count = Int32(0)
        for j0 in cutlass.range(c0, c1, THREADS * 4, unroll=1):
            ss = [mS[(row, min(j0 + u * THREADS + tid, c1 - 1))] for u in range(4)]
            for u in cutlass.range_constexpr(4):
                if (j0 + u * THREADS + tid < c1) & finite(ss[u]):
                    mx = fmax(mx, ss[u])
                    mn = fmin(mn, ss[u])
                    count = count + 1
        for off in cutlass.range_constexpr(5):
            mx = fmax(mx, cute.arch.shuffle_sync_bfly(mx, 1 << off))
            mn = fmin(mn, cute.arch.shuffle_sync_bfly(mn, 1 << off))
            count = count + cute.arch.shuffle_sync_bfly(count, 1 << off)
        if lane == 0:
            sMax[warp] = mx
            sMin[warp] = mn
            sCount[warp] = count
        for b in cutlass.range(tid, NSUB * NUM_BUCKETS, THREADS, unroll=1):
            sHist[b] = Int32(0)
        if tid == 0:
            sTh[0] = Int32(NUM_BUCKETS)
            sTh[1] = Int32(NUM_BUCKETS)
        cute.arch.sync_threads()
        mx = sMax[0]
        mn = sMin[0]
        n = sCount[0]
        for w in cutlass.range_constexpr(1, WARPS):
            mx = fmax(mx, sMax[w])
            mn = fmin(mn, sMin[w])
            n = n + sCount[w]

        o = -mx  # min over x = -score
        hi = -mn  # max over x
        mag = fmax(abs_f32(hi), abs_f32(o))
        span = fmax(fmax(sub_rn(hi, o), mul_rn(mag, Float32(2.0**-8))), Float32(1e-6))
        inv = div_rn(Float32(NUM_BUCKETS - 1), span)
        # The scan's affine map: rounding the subtraction and the product separately could move a score to the
        # adjacent bucket.
        offset = mul_rn(-o, inv)
        neg_inv = -inv

        # pass 2: the histogram, four sub-histograms
        my_hist = (tid // (THREADS // NSUB)) * NUM_BUCKETS
        base = sHist.iterator.toint()
        # Four loads issued before the atomics, which would otherwise keep them one at a time.
        for j0 in cutlass.range(c0, c1, THREADS * 4, unroll=1):
            ss = [mS[(row, min(j0 + u * THREADS + tid, c1 - 1))] for u in range(4)]
            for u in cutlass.range_constexpr(4):
                if (j0 + u * THREADS + tid < c1) & finite(ss[u]):
                    b = f32_to_i32_rz(fma_rn(neg_inv, ss[u], offset))
                    b = min(max(b, Int32(0)), Int32(NUM_BUCKETS - 1))
                    red_add_shared_u32(base + (my_hist + b) * 4, Int32(1))
        cute.arch.sync_threads()
        c = Int32(0)
        for g in cutlass.range_constexpr(NSUB):
            c = c + sHist[g * NUM_BUCKETS + tid]

        # Dual gates from the same histogram:
        #   * mTh: tight slack_topk gate for pruning/refine (keeps pruning survival minimal);
        #   * mThScan: finite-population Bernstein-Hypergeometric scan gate
        #     j' = ceil(j + 4 * sqrt(j * (1 - K/W) * (1 - n/W)) + 4) for full scan, keeping
        #     underfill probability < 1.6e-5 across 32K..1M while tightening short-KV (32K-256K)
        #     candidate counts by 9-10% (cutting Scan stores and Select histogram work).
        window = max(mKe[row] - mKs[row], Int32(0))
        k = topk
        k_scan = topk
        if (slack_topk > Float32(0.0)) & (n < window):
            n_f = i32_to_f32_rn(n)
            win_f = i32_to_f32_rn(window)
            topk_f = i32_to_f32_rn(topk)
            aim = ceil_f32(div_rn(mul_rn(slack_topk, n_f), win_f))
            k = min(max(f32_to_i32_rz(aim), k_min), topk)
            j_f = div_rn(mul_rn(topk_f, n_f), win_f)
            p_rem = fmax(sub_rn(Float32(1.0), div_rn(topk_f, win_f)), Float32(0.0))
            f_rem = fmax(sub_rn(Float32(1.0), div_rn(n_f, win_f)), Float32(0.0))
            var_f = mul_rn(mul_rn(j_f, p_rem), f_rem)
            aim_scan = ceil_f32(add_rn(add_rn(j_f, mul_rn(Float32(4.0), sqrt_rn(var_f))), Float32(4.0)))
            k_scan = max(k, min(max(f32_to_i32_rz(aim_scan), k_min), topk))
        incl, _ = block_inclusive_scan_i32(c, sScan, tid)
        th_addr = sTh.iterator.toint()
        if incl >= k:
            atomic_min_shared_s32(th_addr, tid)
        if incl >= k_scan:
            atomic_min_shared_s32(th_addr + 4, tid)
        cute.arch.sync_threads()
        if tid == 0:
            th = sTh[0]
            th_scan = sTh[1]
            opened = n < k
            # Write mThScan before mTh so aliasing mThScan = mTh preserves mTh.
            mThScan[row] = min(th_scan, Int32(NUM_BUCKETS - 1))
            mTh[row] = min(th, Int32(NUM_BUCKETS - 1))
            out_o = o
            out_inv = inv
            if opened:
                out_o = Float32(0.0)
                out_inv = Float32(0.0)
            mOrigin[row] = out_o
            mInv[row] = out_inv
            mCnt[row] = Int32(0)
