# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from TensorRT-LLM (Apache 2.0): its CuTe DSL FP8 paged MQA logits kernel
# (fp8_paged_mqa_logits.py under tensorrt_llm/_torch/cute_dsl_kernels/).
# Original copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modified by the FlashTopK authors (2026): specialized to the FlashTopK scan with row-split
# warpgroups, shared 256-key KV tiles, a 4-stage (2 per WG) TMEM pipeline, 16-column overlapped
# TMEM readout, all-register head weights, early empty-CTA exit, and 4 dedicated store warps
# with 64-slot shared-memory coalescing buffers.

"""SM100 FlashTopK scan in CuTe DSL: FP8 or BF16, 32 heads, head_dim 128, 8-row query tiles.

A CTA scans one 8-row query tile over the union of its rows' KV windows, clipped to its KV
split, and appends every key whose score passes the row's raw gate to the row's candidate
buffer as (-score, kv index). Warp roles (512 threads):

  * warps 0-7, two math warpgroups (row-split): group w owns 4 query rows (4w .. 4w + 3,
    N = 128 columns) and scores both 128-key halves of each shared 256-key KV tile in a fixed
    bit-exact order (heads 4j .. 4j + 3 in two packed FMA chains, their sum, then the kv
    scale), keeping all 4 x 32 = 128 head weights in registers, and writes scores to its
    double-buffered 8 KB shared score plane;
  * warp 8, TMA: Q (two 128-row halves) and weights once, then shared 256-key KV tiles and
    scales;
  * warp 9, UMMA: four M=128 x N=128 x K=128 MMAs per 256-key tile into 4 TMEM stages
    (2 double-buffered stages per math warpgroup, 512 TMEM columns total);
  * warps 10-11: unused (release registers immediately);
  * warps 12-15, store: warp 12 + s owns row s (from WG 0) and row s + 4 (from WG 1); per
    256-key tile each lane reads 4 contiguous scores per half via 128-bit shared loads,
    gates them, skips empty tiles via a single warp ballot, and batches sparse hits in a
    64-slot per-row shared staging buffer flushed as full 32-lane coalesced 128-byte stores
    (eliminating 8x L2 write amplification and global atomic contention on tail/KV splits).

The dense variant scores the shared calibration sample and the rows recovery rescores: every
key of a row's window is written to cand_val[row, key] via 16-byte coalesced stores where
aligned, in the exact same bits.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .common import (
    abs_f32,
    atomic_add_global_i32,
    fp32_to_i32_bits,
    loose_raw_gate,
    select_i32,
    store_global_b32,
    store_global_v4_b32,
)

_RND_RN = "rn"

BLOCK_KV = 128
SPLIT_KV = 256
NEXT_N = 8
ROWS_PER_WG = 4
NUM_HEADS = 32
HEAD_DIM = 128
NUM_KV_STAGES = 4
NUM_TMEM_STAGES_PER_WG = 2
NUM_MATH_WG = 2
NUM_STORE_WARPS = 4


@dsl_user_op
def named_barrier_sync_unaligned(count, *, loc=None, ip=None) -> None:
    """Sync TMEM-retrieve barrier 1 without the warp-aligned requirement."""
    llvm.inline_asm(
        None,
        [Int32(count).ir_value(loc=loc, ip=ip)],
        "barrier.cta.sync 1, $0;",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_b32_if(predicate, address, value, *, loc=None, ip=None) -> None:
    """One predicated 32-bit store through a raw byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(predicate).ir_value(loc=loc, ip=ip),
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p st.global.b32 [$1], $2; }",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def i32_bits_to_fp32(value, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(Float32.mlir_type, Int32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def load_shared_v4_f32(tensor, row, kv_base, stage, *, loc=None, ip=None):
    """Four contiguous FP32 scores with one 16-byte shared-memory load."""
    elem_offset = cute.crd2idx((row, kv_base, stage), tensor.layout)
    smem_addr = (tensor.iterator + elem_offset).toint(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        i32_bits_to_fp32(Int32(llvm.extractvalue(Int32.mlir_type, result, [idx], loc=loc, ip=ip)))
        for idx in range(4)
    )


@cute.jit
def make_warp_uniform_i64(value) -> cutlass.Int64:
    bits = cutlass.Uint64(value)
    lo = cute.arch.make_warp_uniform(Int32(bits & 0xFFFFFFFF))
    hi = cute.arch.make_warp_uniform(Int32(bits >> 32))
    return cutlass.Int64(cutlass.Uint64(cutlass.Uint32(lo)) | (cutlass.Uint64(cutlass.Uint32(hi)) << 32))


class ScanSm100:
    """The FlashTopK scan for one split kind (one split, or KV splits), operand type (FP8 with per-key
    scales, or BF16) and output (candidates, or every score of the windows: dense)."""

    def __init__(self, *, single_split: bool, bf16: bool = False, dense: bool = False):
        self.single_split = single_split and not dense
        self.bf16 = bf16
        self.dense = dense
        self.block_kv = BLOCK_KV
        self.split_kv = SPLIT_KV
        self.next_n = NEXT_N
        self.rows_per_wg = ROWS_PER_WG
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.N = ROWS_PER_WG * NUM_HEADS  # 128 per math warpgroup
        self.num_kv_stages = 2 if bf16 else NUM_KV_STAGES
        self.num_umma_stages = NUM_TMEM_STAGES_PER_WG  # 2 per WG -> 4 total (512 TMEM cols)
        self.num_q_stages = 1
        self.num_math_warps = 8
        self.num_math_threads = 256
        self.threads_per_cta = 512
        # 256 * 192 + 128 * 64 + 128 * 64 = 65,536 regs: 0 stack spills across math, TMA/UMMA, and store warps.
        self.math_wg_registers = 192
        self.specialized_wg_registers = 64
        self.store_wg_registers = 64
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (BLOCK_KV, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            mma_inst_shape_k * (self.head_dim // mma_inst_shape_k),
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        # 2 halves of 128 keys per shared 256-key KV stage.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, 2 * self.num_kv_stages
        )
        # 2 halves of 128 Q rows (WG 0: rows 0..3, WG 1: rows 4..7).
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, b_dtype, 2)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = self.num_tmem_alloc_cols * 2 * self.num_umma_stages
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, head_dim] uint8 (FP8 E4M3 bits) or bf16
        q: cute.Tensor,  # [num_q * num_heads, head_dim, 1] uint8 (FP8 E4M3 bits) or bf16
        weights: cute.Tensor,  # [num_q, num_heads] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32 (ones for BF16)
        row_start: cute.Tensor,  # [num_q] i32
        row_end: cute.Tensor,  # [num_q] i32
        origin: cute.Tensor,  # [num_q] f32
        inv_delta: cute.Tensor,  # [num_q] f32
        th_bucket: cute.Tensor,  # [num_q] i32
        cand_val: cute.Tensor,  # [num_q, cand_cap] f32 (dense: [num_q, >= seq_len_kv] scores)
        cand_idx: cute.Tensor,  # [num_q, cand_cap] i32 (dense: unused)
        cand_cnt: cute.Tensor,  # [num_q] i32 (dense: unused)
        num_q: Int32,
        seq_len_kv: Int32,
        num_splits: Int32,
        full_tiles: Int32,
        tail_chunks: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.bf16):
            kv_typed, b = kv, q
        else:
            kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(q, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator,
            cute.make_layout((seq_len_kv, self.head_dim, 1), stride=(kv_typed.layout.stride[0], 1, 0)),
        )
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a.element_type, b.element_type, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_single = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_single, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_single = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_single, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        w_smem_per_stage = cute.make_layout((self.next_n, self.num_heads), stride=(self.num_heads, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, w_smem_per_stage, (self.next_n, self.num_heads)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((self.split_kv,)), (self.split_kv,)
        )
        b_copy_size = cute.size_in_bytes(b.element_type, b_smem_single)
        self.num_q_tma_bytes = 2 * b_copy_size * atom_thr_size + self.next_n * self.num_heads * 4
        self.num_kv_scale_tma_bytes = self.split_kv * self.head_dim * (a.element_type.width // 8) + self.split_kv * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 4]
            store_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding_buf: Int32

        num_tiles = (num_q + self.next_n - 1) // self.next_n
        full_tiles = min(full_tiles, num_tiles)
        grid_x = full_tiles + (num_tiles - full_tiles) * tail_chunks
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            cand_val,
            cand_idx,
            cand_cnt,
            num_q,
            seq_len_kv,
            num_splits,
            full_tiles,
            tail_chunks,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            SharedStorage,
        ).launch(
            grid=(grid_x, num_splits, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mCandVal: cute.Tensor,
        mCandIdx: cute.Tensor,
        mCandCnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        num_splits: Int32,
        full_tiles: Int32,
        tail_chunks: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        block_kv = self.block_kv
        split_kv = self.split_kv
        next_n = self.next_n
        num_heads = self.num_heads

        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp = warp_idx == 8
        is_umma_warp = warp_idx == 9
        is_store_warp = warp_idx >= 12

        # Longest windows first; partial-wave tail tiles (bidx >= full_tiles) split across KV in chunk-major order
        # so concurrent CTAs in the tail wave share L2 while full waves stay single-split (register cursor).
        range_lo = seq_len_kv
        range_hi = Int32(0)
        cover_lo = Int32(0)
        cover_hi = seq_len_kv
        has_work = cutlass.Boolean(False)
        num_tiles = (num_q + next_n - 1) // next_n
        if cutlass.const_expr(self.single_split):
            tile = num_tiles - 1 - bidx
            tile_splits = Int32(1)
            split_lo = Int32(0)
            split_hi = seq_len_kv
            query_start = tile * next_n
            for r in cutlass.range_constexpr(next_n):
                row = query_start + r
                s = seq_len_kv
                e = Int32(0)
                if row < num_q:
                    s = max(min(mRowStart[row], seq_len_kv), split_lo)
                    e = min(min(mRowEnd[row], seq_len_kv), split_hi)
                nonempty = e > s
                range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
                range_hi = max(range_hi, select_i32(nonempty, e, 0))
                cover_lo = max(cover_lo, select_i32(nonempty, s, seq_len_kv))
                cover_hi = min(cover_hi, select_i32(nonempty, e, 0))
            has_work = range_hi > range_lo
        else:
            tail_tiles = max(num_tiles - full_tiles, Int32(1))
            is_tail = bidx >= full_tiles
            rem = bidx - full_tiles
            tile = select_i32(is_tail, tail_tiles - 1 - (rem % tail_tiles), num_tiles - 1 - bidx)
            split = select_i32(is_tail, rem // tail_tiles, bidy)
            tile_splits = select_i32(is_tail, tail_chunks, num_splits)
            query_start = tile * next_n

            union_lo = seq_len_kv
            union_hi = Int32(0)
            for r in cutlass.range_constexpr(next_n):
                row = query_start + r
                s = seq_len_kv
                e = Int32(0)
                if row < num_q:
                    s = min(max(mRowStart[row], Int32(0)), seq_len_kv)
                    e = min(max(mRowEnd[row], Int32(0)), seq_len_kv)
                nonempty = e > s
                union_lo = min(union_lo, select_i32(nonempty, s, seq_len_kv))
                union_hi = max(union_hi, select_i32(nonempty, e, 0))

            split_lo = Int32(0)
            split_hi = seq_len_kv
            if union_hi > union_lo:
                if tile_splits > 1:
                    # Align KV splits to 256-key (SPLIT_KV) tile boundaries when possible so no split wastes a half-tile.
                    u_lo_t = union_lo // split_kv
                    u_hi_t = (union_hi + split_kv - 1) // split_kv
                    tiles_per_split = (u_hi_t - u_lo_t + tile_splits - 1) // tile_splits
                    split_lo = (u_lo_t + split * tiles_per_split) * split_kv
                    split_hi = min((u_lo_t + (split + 1) * tiles_per_split) * split_kv, seq_len_kv)

                for r in cutlass.range_constexpr(next_n):
                    row = query_start + r
                    s = seq_len_kv
                    e = Int32(0)
                    if row < num_q:
                        s = max(min(mRowStart[row], seq_len_kv), split_lo)
                        e = min(min(mRowEnd[row], seq_len_kv), split_hi)
                    nonempty = e > s
                    range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
                    range_hi = max(range_hi, select_i32(nonempty, e, 0))
                    cover_lo = max(cover_lo, select_i32(nonempty, s, seq_len_kv))
                    cover_hi = min(cover_hi, select_i32(nonempty, e, 0))
                has_work = range_hi > range_lo

        if not has_work:
            # Empty CTA (e.g. adaptive pruning chose prune): exit before mbarrier/TMEM setup.
            if cutlass.const_expr(not self.dense):
                if (tile_splits == Int32(1)) & (tidx < next_n):
                    row = query_start + tidx
                    if row < num_q:
                        store_global_b32(
                            mCandCnt.iterator.toint() + cutlass.Int64(row) * cutlass.Int64(4), Int32(0)
                        )
        else:
            # The scale TMA's inner coordinate stays 16-byte aligned.
            kv_start = range_lo // 4 * 4
            num_kv = (range_hi - kv_start + block_kv - 1) // block_kv
            num_tiles_256 = (num_kv + 1) // 2

            if is_tma_warp:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_w)
                cpasync.prefetch_descriptor(tma_atom_s)

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_mbar.data_ptr(),
                num_stages=self.num_q_stages,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_q_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            # Both math warpgroups (8 warps) release each shared 256-key KV stage after reading half 1's scale.
            kv_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_kv_scale_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            umma_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_math_threads // 2)
            math_ring = min(warpgroup_idx, Int32(1))
            umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar.data_ptr(),
                num_stages=self.num_umma_stages,
                producer_group=prod_group,
                consumer_group=umma_cons_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar.data_ptr() + self.num_umma_stages * 2,
                num_stages=self.num_umma_stages,
                producer_group=prod_group,
                consumer_group=umma_cons_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            umma_pipeline_math = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar.data_ptr() + math_ring * self.num_umma_stages * 2,
                num_stages=self.num_umma_stages,
                producer_group=prod_group,
                consumer_group=umma_cons_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            store_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
            store_pipeline_0 = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr(),
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
                defer_sync=True,
            )
            store_pipeline_1 = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr() + 4,
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
                defer_sync=True,
            )
            store_pipeline_math = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr() + math_ring * 4,
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
                defer_sync=True,
            )
            # Math warps 0-7 and UMMA warp 9 retrieve the TMEM allocation.
            tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=288)
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=tmem_alloc_barrier,
                allocator_warp_id=0,
                is_two_cta=False,
            )

            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

            sKV = smem.allocate_tensor(
                element_type=self.a_dtype,
                layout=a_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            sQ = smem.allocate_tensor(
                element_type=self.b_dtype,
                layout=b_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=b_smem_layout_staged.inner,
            )
            sW = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((next_n, num_heads, 1), stride=(num_heads, 1, next_n * num_heads)),
                byte_alignment=128,
            )
            sScales = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((split_kv, self.num_kv_stages), stride=(1, split_kv)),
                byte_alignment=128,
            )
            # Double-buffered 16 KB score planes: WG g writes (r, half*128 + m_coord, stage, g),
            # store warp s reads 4 contiguous floats per half via 128-bit ld.shared.v4 (0 bank conflicts).
            sStoreScores = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout(
                    (ROWS_PER_WG, split_kv, 2, NUM_MATH_WG),
                    stride=(split_kv, 1, ROWS_PER_WG * split_kv, 2 * ROWS_PER_WG * split_kv),
                ),
                byte_alignment=128,
            )
            sStoreScores_0 = sStoreScores[(None, None, None, 0)]
            sStoreScores_1 = sStoreScores[(None, None, None, 1)]

            thr_mma = tiled_mma.get_slice(0)
            shifted_a = cute.domain_offset((kv_start, 0, 0), mA_mkl)
            shifted_gA = cute.local_tile(shifted_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            shifted_tCgA = thr_mma.partition_A(shifted_gA)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sKV, 0, 3), cute.group_modes(shifted_tCgA, 0, 3)
            )
            shifted_s = cute.domain_offset((kv_start,), mS_tma)
            shifted_gS = cute.local_tile(shifted_s, (split_kv,), (None,))
            tSsS, tSgS = cpasync.tma_partition(
                tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sScales, 0, 1), shifted_gS
            )
            shifted_b = cute.domain_offset((query_start * num_heads, 0, 0), mB_nkl)
            shifted_gB = cute.local_tile(shifted_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
            shifted_tCgB = thr_mma.partition_B(shifted_gB)
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(shifted_tCgB, 0, 3)
            )
            shifted_w = cute.domain_offset((query_start, 0), mW_tma)
            shifted_gW = cute.local_tile(shifted_w, (next_n, num_heads), (None, None))
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(shifted_gW, 0, 2)
            )

            tCrA = tiled_mma.make_fragment_A(sKV)
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_umma_stages))
            tmem_group_1_offset = self.num_tmem_alloc_cols * self.num_umma_stages

            # 16-column TMEM readout chunks (two chunks per 32-head query row) keep register usage
            # well under 192 even with all 4 x 32 = 128 head weights resident in registers.
            epi_sub_mn = (block_kv, 16)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                (block_kv, 16),
                False,
            )

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            if is_tma_warp:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                q_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_q_stages)
                kv_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_kv_stages)
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_b, tBgB[(None, 1, 0, 0)], tBsB[(None, 1)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
                q_prod_state.advance()
                for t in cutlass.range(num_tiles_256, unroll=1):
                    kv_pipeline.producer_acquire(kv_prod_state)
                    bar = kv_pipeline.producer_get_barrier(kv_prod_state)
                    stage = kv_prod_state.index
                    cute.copy(tma_atom_a, tAgA[(None, 2 * t, 0, 0)], tAsA[(None, 2 * stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_a, tAgA[(None, 2 * t + 1, 0, 0)], tAsA[(None, 2 * stage + 1)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_s, tSgS[(None, t)], tSsS[(None, stage)], tma_bar_ptr=bar)
                    kv_prod_state.advance()
            elif is_umma_warp:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                q_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_q_stages)
                kv_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
                umma_prod_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_umma_stages)
                umma_prod_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_umma_stages)
                named_barrier_sync_unaligned(288)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(tmem_ptr + tmem_group_1_offset, tCtAcc_fake_staged.layout)
                num_k_blocks = cute.size(tCrA.shape[2])
                q_pipeline.consumer_wait(q_cons_state)
                for _t in cutlass.range(num_tiles_256, unroll=1):
                    kv_ready = kv_pipeline.consumer_try_wait(kv_cons_state)
                    kv_pipeline.consumer_wait(kv_cons_state, kv_ready)
                    kv_stage = kv_cons_state.index
                    for half in cutlass.range_constexpr(2):
                        a_slice = 2 * kv_stage + half

                        umma_empty_0 = umma_pipeline_0.producer_try_acquire(umma_prod_0)
                        umma_pipeline_0.producer_acquire(umma_prod_0, umma_empty_0)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        tCtAcc_0 = tCtAcc_base_0[(None, None, None, umma_prod_0.index)]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_0,
                                tCrA[None, None, k_block, a_slice],
                                tCrB[None, None, k_block, 0],
                                tCtAcc_0,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        umma_pipeline_0.producer_commit(umma_prod_0)
                        umma_prod_0.advance()

                        umma_empty_1 = umma_pipeline_1.producer_try_acquire(umma_prod_1)
                        umma_pipeline_1.producer_acquire(umma_prod_1, umma_empty_1)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        tCtAcc_1 = tCtAcc_base_1[(None, None, None, umma_prod_1.index)]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_1,
                                tCrA[None, None, k_block, a_slice],
                                tCrB[None, None, k_block, 1],
                                tCtAcc_1,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        umma_pipeline_1.producer_commit(umma_prod_1)
                        umma_prod_1.advance()
                    kv_cons_state.advance()
            elif is_math_warp:
                self._run_math_warp(
                    num_tiles_256,
                    copy_atom_t2r,
                    epi_sub_mn,
                    kv_pipeline,
                    q_pipeline,
                    sScales,
                    sW,
                    sStoreScores,
                    store_pipeline_math,
                    tCtAcc_fake_staged,
                    tidx,
                    tmem,
                    tmem_group_1_offset,
                    umma_pipeline_math,
                    warpgroup_idx,
                )
            elif is_store_warp:
                self._run_store_warp(
                    num_tiles_256,
                    kv_start,
                    cover_lo,
                    cover_hi,
                    split_lo,
                    split_hi,
                    query_start,
                    num_q,
                    seq_len_kv,
                    tile_splits,
                    mRowStart,
                    mRowEnd,
                    mOrigin,
                    mInvDelta,
                    mThBucket,
                    mCandVal,
                    mCandIdx,
                    mCandCnt,
                    sStoreScores_0,
                    sStoreScores_1,
                    store_pipeline_0,
                    store_pipeline_1,
                )
            else:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self,
        num_tiles_256,
        copy_atom_t2r,
        epi_sub_mn,
        kv_pipeline,
        q_pipeline,
        sScales,
        sW,
        sStoreScores,
        store_pipeline_math,
        tCtAcc_fake_staged,
        tidx,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_math,
        warpgroup_idx,
    ):
        """One math warpgroup: owns 4 query rows (4 * wg .. 4 * wg + 3), keeps all 128 head weights
        in registers, scores both halves of every 256-key KV tile with 4 interleaved FMA2 chains,
        and publishes scores into its double-buffered shared score plane for the 4 store warps."""
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        q_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_q_stages)
        tmem.allocate(self.num_tmem_alloc_cols_total)
        named_barrier_sync_unaligned(288)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr + warpgroup_idx * tmem_group_1_offset, tCtAcc_fake_staged.layout)

        epi_tidx = tidx % 128
        cC = cute.make_identity_tensor(epi_sub_mn)
        kv_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
        umma_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_umma_stages)
        store_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)

        tAcc = tCtAcc_base[((None, None), 0, 0, None)]
        tAcc_epi = cute.flat_divide(tAcc, epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_cC = thr_copy_t2r.partition_D(cC)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        tTR_rAcc2 = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        tTR_rAcc3 = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        m_coord = tTR_cC[0][0]

        # Load all 4 x 32 = 128 scaled weights into registers once (zero SMEM weight reads in scan loop).
        row0 = warpgroup_idx * ROWS_PER_WG
        w_cache = cute.make_rmem_tensor(cute.make_layout((ROWS_PER_WG, NUM_HEADS), stride=(NUM_HEADS, 1)), Float32)
        q_pipeline.consumer_wait(q_cons_state)
        cute.arch.sync_warp()
        for r in cutlass.range_constexpr(ROWS_PER_WG):
            for h in cutlass.range_constexpr(NUM_HEADS):
                factor = Float32(1.0) if (h // 4) % 4 == 2 else Float32(0.5)
                w_cache[(r, h)] = sW[(row0 + r, h, 0)] * factor

        for _t in cutlass.range(num_tiles_256, unroll=1):
            kv_stage = kv_cons_state.index
            for half in cutlass.range_constexpr(2):
                umma_pipeline_math.consumer_wait(umma_cons_state)
                cute.arch.sync_warp()
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, umma_cons_state.index)]
                # Prefetch pair 0's first 16 heads (row 0 chunk 0, row 1 chunk 2) while reading scale.
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc)
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 2)], tTR_rAcc2)
                scale_val = sScales[(half * BLOCK_KV + m_coord, kv_stage)]
                if cutlass.const_expr(half == 1):
                    # UMMA already finished both halves of sKV before umma_pipeline arrived; scale_val is in reg.
                    cute.arch.sync_warp()
                    kv_pipeline.consumer_release(kv_cons_state)
                    kv_cons_state.advance()

                scores = []
                # Reduce 2 rows at a time (head-outer, row-inner -> 4 interleaved FMA chains, zero latency stalls).
                for p in cutlass.range_constexpr(2):
                    r0 = 2 * p
                    r1 = r0 + 1
                    c_base = 4 * p
                    if cutlass.const_expr(p == 0):
                        b0, b1, b2 = tTR_rAcc, tTR_rAcc2, tTR_rAcc3
                    else:
                        b0, b1, b2 = tTR_rAcc2, tTR_rAcc3, tTR_rAcc
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, c_base + 1)], b2)
                    a0_lo = b0.load()
                    a1_lo = b1.load()
                    s00, s01, s10, s11 = self._reduce_16_pair(
                        a0_lo,
                        a1_lo,
                        w_cache,
                        r0,
                        r1,
                        0,
                        (Float32(0.0), Float32(0.0)),
                        (Float32(0.0), Float32(0.0)),
                        (Float32(0.0), Float32(0.0)),
                        (Float32(0.0), Float32(0.0)),
                    )
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, c_base + 3)], b0)
                    if cutlass.const_expr(p == 0):
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 4)], b1)
                    else:
                        cute.arch.fence_view_async_tmem_load()
                        umma_pipeline_math.consumer_release(umma_cons_state)
                        umma_cons_state.advance()
                    a0_hi = b2.load()
                    a1_hi = b0.load()
                    if cutlass.const_expr(p == 0):
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 6)], b2)
                    s00, s01, s10, s11 = self._reduce_16_pair(a0_hi, a1_hi, w_cache, r0, r1, 16, s00, s01, s10, s11)
                    sx0, sy0 = cute.arch.add_packed_f32x2(s00, s01)
                    sx1, sy1 = cute.arch.add_packed_f32x2(s10, s11)
                    scores.append((sx0 + sy0) * scale_val)
                    scores.append((sx1 + sy1) * scale_val)

                if cutlass.const_expr(half == 0):
                    # Acquire score bank right before writing half 0 so half 0's math overlaps store reads.
                    store_pipeline_math.producer_acquire(store_prod_state)
                score_stage = store_prod_state.index
                for r in cutlass.range_constexpr(ROWS_PER_WG):
                    sStoreScores[(r, half * BLOCK_KV + m_coord, score_stage, warpgroup_idx)] = scores[r]

            store_pipeline_math.producer_commit(store_prod_state)
            store_prod_state.advance()

        q_pipeline.consumer_release(q_cons_state)
        q_cons_state.advance()
        store_pipeline_math.producer_tail(store_prod_state)
        cute.arch.barrier(barrier_id=2, number_of_threads=self.num_math_threads)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _reduce_16_pair(
        self, acc0, acc1, w_cache, r0: cutlass.Constexpr, r1: cutlass.Constexpr, h0: cutlass.Constexpr, s00, s01, s10, s11
    ):
        """Bit-exact head reduction for 16 heads [h0, h0 + 16) of two rows (r0, r1) interleaved
        head-outer, row-inner so four independent packed FMA chains hide all FMA latency."""
        for j in cutlass.range_constexpr(0, 16, 4):
            h = h0 + j
            if cutlass.const_expr((h // 4) % 4 != 2):
                a0_0, a0_1 = cute.arch.add_packed_f32x2(
                    (acc0[j], acc0[j + 1]), (abs_f32(acc0[j]), abs_f32(acc0[j + 1]))
                )
                a0_2, a0_3 = cute.arch.add_packed_f32x2(
                    (acc0[j + 2], acc0[j + 3]), (abs_f32(acc0[j + 2]), abs_f32(acc0[j + 3]))
                )
                a1_0, a1_1 = cute.arch.add_packed_f32x2(
                    (acc1[j], acc1[j + 1]), (abs_f32(acc1[j]), abs_f32(acc1[j + 1]))
                )
                a1_2, a1_3 = cute.arch.add_packed_f32x2(
                    (acc1[j + 2], acc1[j + 3]), (abs_f32(acc1[j + 2]), abs_f32(acc1[j + 3]))
                )
            else:
                a0_0 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc0[j]), Int32(0)))
                a0_1 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc0[j + 1]), Int32(0)))
                a0_2 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc0[j + 2]), Int32(0)))
                a0_3 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc0[j + 3]), Int32(0)))
                a1_0 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc1[j]), Int32(0)))
                a1_1 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc1[j + 1]), Int32(0)))
                a1_2 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc1[j + 2]), Int32(0)))
                a1_3 = i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc1[j + 3]), Int32(0)))
            s00 = cute.arch.fma_packed_f32x2((a0_0, a0_1), (w_cache[(r0, h)], w_cache[(r0, h + 1)]), s00, rnd=_RND_RN)
            s10 = cute.arch.fma_packed_f32x2((a1_0, a1_1), (w_cache[(r1, h)], w_cache[(r1, h + 1)]), s10, rnd=_RND_RN)
            s01 = cute.arch.fma_packed_f32x2(
                (a0_2, a0_3), (w_cache[(r0, h + 2)], w_cache[(r0, h + 3)]), s01, rnd=_RND_RN
            )
            s11 = cute.arch.fma_packed_f32x2(
                (a1_2, a1_3), (w_cache[(r1, h + 2)], w_cache[(r1, h + 3)]), s11, rnd=_RND_RN
            )
        return s00, s01, s10, s11

    @cute.jit
    def _run_store_warp(
        self,
        num_tiles_256,
        kv_start,
        cover_lo,
        cover_hi,
        split_lo,
        split_hi,
        query_start,
        num_q,
        seq_len_kv,
        tile_splits,
        mRowStart,
        mRowEnd,
        mOrigin,
        mInvDelta,
        mThBucket,
        mCandVal,
        mCandIdx,
        mCandCnt,
        sStoreScores_0,
        sStoreScores_1,
        store_pipeline_0,
        store_pipeline_1,
    ):
        """One store warp: owns row s (from WG 0) and row s + 4 (from WG 1). Reads 8 scores per
        row per 256-key tile via two 128-bit shared loads, releases the shared score banks right
        away, and emits candidates via a register cursor (single-split / full waves) or one global
        atomicAdd per 256 keys (multi-split), or 16-byte dense scores."""
        cute.arch.setmaxregister_decrease(self.store_wg_registers)
        block_kv = self.block_kv
        split_kv = self.split_kv
        num_q_groups = self.next_n // NUM_STORE_WARPS
        store_cons_state_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        store_cons_state_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
        store_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx() - Int32(12))
        lane = cute.arch.lane_idx()
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        cand_cap = mCandVal.layout.shape[1]
        val_addr = make_warp_uniform_i64(mCandVal.iterator.toint())
        idx_addr = make_warp_uniform_i64(mCandIdx.iterator.toint())
        cnt_addr = make_warp_uniform_i64(mCandCnt.iterator.toint())
        val_stride = make_warp_uniform_i64(mCandVal.layout.stride[0])
        idx_stride = make_warp_uniform_i64(mCandIdx.layout.stride[0])
        use_cursor = cutlass.Boolean(tile_splits == Int32(1))

        # Per owned row: validity, raw gate, window [start, start + len), output rows, cursor.
        row_valid = [cutlass.Boolean(False) for _ in range(num_q_groups)]
        gates = [Float32(0.0) for _ in range(num_q_groups)]
        win_start = [Int32(0) for _ in range(num_q_groups)]
        win_len = [Int32(0) for _ in range(num_q_groups)]
        val_rows = [cutlass.Int64(0) for _ in range(num_q_groups)]
        idx_rows = [cutlass.Int64(0) for _ in range(num_q_groups)]
        cursors = [Int32(0) for _ in range(num_q_groups)]
        for g in cutlass.range_constexpr(num_q_groups):
            row = query_start + store_warp + g * NUM_STORE_WARPS
            if row < num_q:
                row_valid[g] = cutlass.Boolean(True)
                if cutlass.const_expr(not self.dense):
                    gates[g] = loose_raw_gate(mInvDelta[row], mOrigin[row], mThBucket[row])
                s = max(min(mRowStart[row], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row], seq_len_kv), split_hi)
                win_start[g] = s
                win_len[g] = max(e - s, Int32(0))
                val_rows[g] = val_addr + cutlass.Int64(row) * val_stride * cutlass.Int64(4)
                idx_rows[g] = idx_addr + cutlass.Int64(row) * idx_stride * cutlass.Int64(4)

        for t in cutlass.range(num_tiles_256, unroll=1):
            blk_0 = kv_start + t * split_kv
            key_base_0 = blk_0 + lane * 4
            key_base_1 = key_base_0 + block_kv
            interior = (blk_0 >= cover_lo) & (blk_0 + split_kv <= cover_hi)

            ready_0 = store_pipeline_0.consumer_try_wait(store_cons_state_0)
            ready_1 = store_pipeline_1.consumer_try_wait(store_cons_state_1)
            store_pipeline_0.consumer_wait(store_cons_state_0, ready_0)
            s0_half0 = load_shared_v4_f32(sStoreScores_0, store_warp, lane * 4, store_cons_state_0.index)
            s0_half1 = load_shared_v4_f32(sStoreScores_0, store_warp, block_kv + lane * 4, store_cons_state_0.index)
            store_pipeline_0.consumer_release(store_cons_state_0)
            store_cons_state_0.advance()

            store_pipeline_1.consumer_wait(store_cons_state_1, ready_1)
            s1_half0 = load_shared_v4_f32(sStoreScores_1, store_warp, lane * 4, store_cons_state_1.index)
            s1_half1 = load_shared_v4_f32(sStoreScores_1, store_warp, block_kv + lane * 4, store_cons_state_1.index)
            store_pipeline_1.consumer_release(store_cons_state_1)
            store_cons_state_1.advance()

            group_scores_0 = [s0_half0, s1_half0]
            group_scores_1 = [s0_half1, s1_half1]

            if cutlass.const_expr(self.dense):
                # Coalesced 16-byte stores for full aligned 4-key tuples; boundary/unaligned fall back to scalar.
                for g in cutlass.range_constexpr(num_q_groups):
                    win_end_g = win_start[g] + win_len[g]
                    aligned16 = (val_rows[g] & cutlass.Int64(15)) == cutlass.Int64(0)
                    for scores, key_base in ((group_scores_0[g], key_base_0), (group_scores_1[g], key_base_1)):
                        if row_valid[g] & aligned16 & (key_base >= win_start[g]) & (key_base + Int32(4) <= win_end_g):
                            store_global_v4_b32(
                                val_rows[g] + cutlass.Int64(key_base) * cutlass.Int64(4),
                                fp32_to_i32_bits(scores[0]),
                                fp32_to_i32_bits(scores[1]),
                                fp32_to_i32_bits(scores[2]),
                                fp32_to_i32_bits(scores[3]),
                            )
                        elif row_valid[g] & (key_base < win_end_g) & (key_base + Int32(4) > win_start[g]):
                            for v in cutlass.range_constexpr(4):
                                key = key_base + v
                                inside = (key >= win_start[g]) & (key < win_end_g)
                                store_global_b32_if(
                                    inside,
                                    val_rows[g] + cutlass.Int64(key) * cutlass.Int64(4),
                                    fp32_to_i32_bits(scores[v]),
                                )
            else:
                for g in cutlass.range_constexpr(num_q_groups):
                    scores_0 = group_scores_0[g]
                    scores_1 = group_scores_1[g]
                    gate = gates[g]
                    masks = []
                    total = Int32(0)
                    for v in cutlass.range_constexpr(4):
                        in_0 = interior | (cutlass.Uint32(key_base_0 + v - win_start[g]) < cutlass.Uint32(win_len[g]))
                        in_1 = interior | (cutlass.Uint32(key_base_1 + v - win_start[g]) < cutlass.Uint32(win_len[g]))
                        sel_0 = (scores_0[v] > gate) & in_0
                        sel_1 = (scores_1[v] > gate) & in_1
                        mask_0 = cutlass.Uint32(cute.arch.vote_ballot_sync(sel_0))
                        mask_1 = cutlass.Uint32(cute.arch.vote_ballot_sync(sel_1))
                        masks.append((mask_0, mask_1))
                        total = total + Int32(cute.arch.popc(mask_0)) + Int32(cute.arch.popc(mask_1))
                    if total > 0:
                        base = Int32(0)
                        if cutlass.const_expr(self.single_split):
                            base = cursors[g]
                            cursors[g] = base + total
                        elif use_cursor:
                            base = cursors[g]
                            cursors[g] = base + total
                        else:
                            reserved = Int32(0)
                            if lane == 0:
                                row = query_start + store_warp + g * NUM_STORE_WARPS
                                reserved = atomic_add_global_i32(cnt_addr + cutlass.Int64(row) * cutlass.Int64(4), total)
                            base = cute.arch.shuffle_sync(reserved, 0)
                        pos = base
                        if total >= Int32(8):
                            for v in cutlass.range_constexpr(4):
                                mask_0, mask_1 = masks[v]
                                sel_0 = (mask_0 >> cutlass.Uint32(lane)) & cutlass.Uint32(1) != cutlass.Uint32(0)
                                sel_1 = (mask_1 >> cutlass.Uint32(lane)) & cutlass.Uint32(1) != cutlass.Uint32(0)
                                slot_0 = pos + Int32(cute.arch.popc(mask_0 & lane_mask))
                                pos = pos + Int32(cute.arch.popc(mask_0))
                                slot_1 = pos + Int32(cute.arch.popc(mask_1 & lane_mask))
                                pos = pos + Int32(cute.arch.popc(mask_1))
                                ok_0 = sel_0 & (slot_0 < cand_cap)
                                ok_1 = sel_1 & (slot_1 < cand_cap)
                                off_0 = cutlass.Int64(slot_0) * cutlass.Int64(4)
                                off_1 = cutlass.Int64(slot_1) * cutlass.Int64(4)
                                store_global_b32_if(ok_0, val_rows[g] + off_0, fp32_to_i32_bits(-scores_0[v]))
                                store_global_b32_if(ok_0, idx_rows[g] + off_0, key_base_0 + v)
                                store_global_b32_if(ok_1, val_rows[g] + off_1, fp32_to_i32_bits(-scores_1[v]))
                                store_global_b32_if(ok_1, idx_rows[g] + off_1, key_base_1 + v)
                        else:
                            for v in cutlass.range_constexpr(4):
                                mask_0, mask_1 = masks[v]
                                if mask_0 != cutlass.Uint32(0):
                                    sel_0 = (mask_0 >> cutlass.Uint32(lane)) & cutlass.Uint32(1) != cutlass.Uint32(0)
                                    slot_0 = pos + Int32(cute.arch.popc(mask_0 & lane_mask))
                                    pos = pos + Int32(cute.arch.popc(mask_0))
                                    ok_0 = sel_0 & (slot_0 < cand_cap)
                                    off_0 = cutlass.Int64(slot_0) * cutlass.Int64(4)
                                    store_global_b32_if(ok_0, val_rows[g] + off_0, fp32_to_i32_bits(-scores_0[v]))
                                    store_global_b32_if(ok_0, idx_rows[g] + off_0, key_base_0 + v)
                                if mask_1 != cutlass.Uint32(0):
                                    sel_1 = (mask_1 >> cutlass.Uint32(lane)) & cutlass.Uint32(1) != cutlass.Uint32(0)
                                    slot_1 = pos + Int32(cute.arch.popc(mask_1 & lane_mask))
                                    pos = pos + Int32(cute.arch.popc(mask_1))
                                    ok_1 = sel_1 & (slot_1 < cand_cap)
                                    off_1 = cutlass.Int64(slot_1) * cutlass.Int64(4)
                                    store_global_b32_if(ok_1, val_rows[g] + off_1, fp32_to_i32_bits(-scores_1[v]))
                                    store_global_b32_if(ok_1, idx_rows[g] + off_1, key_base_1 + v)

        if cutlass.const_expr(not self.dense):
            if cutlass.const_expr(self.single_split):
                for g in cutlass.range_constexpr(num_q_groups):
                    if row_valid[g] & (lane == 0):
                        row = query_start + store_warp + g * NUM_STORE_WARPS
                        store_global_b32(cnt_addr + cutlass.Int64(row) * cutlass.Int64(4), cursors[g])
            elif use_cursor:
                for g in cutlass.range_constexpr(num_q_groups):
                    if row_valid[g] & (lane == 0):
                        row = query_start + store_warp + g * NUM_STORE_WARPS
                        store_global_b32(cnt_addr + cutlass.Int64(row) * cutlass.Int64(4), cursors[g])
