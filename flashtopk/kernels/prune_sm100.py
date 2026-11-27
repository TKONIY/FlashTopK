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
# Modified by the FlashTopK authors (2026):
#   * PruneSm100 (16-col, R=14): row-split warpgroups over shared 256-key KV tiles (4 KV stages),
#     3-stage TMEM ring (N=144, 3 x 144 = 432 <= 512 cols), double-buffered TMEM readout,
#     all-register retained weights and per-pair metadata, and lossless linear-term upper-bound
#     reduction (zero ReLU instructions on retained heads; L2 quantization residual folded into
#     `allowance` in plan.py so the final top-k stays 100% bit-exact).
#   * Prune24Sm100 (24-col, R=22): row-split warpgroups (16 rows/CTA, 8 rows/WG, N=208) over
#     shared 128-key KV tiles (8 KV stages FP8 / 3 BF16, halving KV HBM traffic vs 8-row tiles),
#     lossless linear-term upper-bound reduction (0 ReLU ops), double-buffered 16-col TMEM
#     readout, 128 register-cached retained weights, and early KV release.

"""SM100 FlashTopK key pruning in CuTe DSL (FP8 or BF16, 32 heads): the score upper bound,
the first of pruning's two kernels."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .common import abs_f32, fmax, loose_raw_gate, select_i32, store_global_b32
from .scan_sm100 import _RND_RN, named_barrier_sync_unaligned

BLOCK_KV = 128
SPLIT_KV = 256
COLS = 16
COLS_24 = 24
HEAD_DIM = 128
NUM_KV_STAGES = 4
NUM_KV_STAGES_24 = 8
NUM_W_REG_24 = 26  # first 13 retained heads (4 pairs x 26 = 104 floats) -> 0 stack spill at 224 regs
NUM_EPI_SUBTILES = 4
NUM_MATH_WG = 2
TMEM_TOTAL_COLS = 512


class PruneSm100:
    """16-col SM100 score-upper-bound pruning kernel (R=14 retained heads, 16 rows/CTA).

    For each row, the retained heads' ReLU sum is split algebraically as

        sum_{c < 14} wt_c relu(x_c) = sum_{c < 14} (wt_c / 2) |x_c| + q_lin . k,

    where q_lin = sum_{c < 14} (wt_c / 2) q_c is quantized into two extra operand rows (lin_hi,
    lin_lo) computed on the Tensor Cores, and its L2 quantization residual is added to `allowance`
    via Cauchy-Schwarz in plan.py so the upper bound remains 100% sound (zero false negatives).
    In the math warps, |x_c| folds into FFMA2's free `.abs` operand modifier (0 ReLU ops).

    Row-split warpgroup layout over shared 256-key KV tiles:
      * WG w owns 8 rows (one complete 8-row refine group), N = 144 (128 main + 16 linear cols),
        3-stage TMEM ring (3 x 144 = 432 <= 512 cols);
      * Warp roles (384 threads): warps 0-7 math, warp 8 TMA, warp 9 UMMA, warps 10-11 unused.
    """

    def __init__(self, *, bf16: bool = False, cols: int = COLS):
        assert cols == COLS, f"PruneSm100 handles 16 columns (use Prune24Sm100 for 24), got {cols}"
        self.bf16 = bf16
        self.cols = COLS
        self.retained = COLS - 2  # 14
        self.w_cols = 20  # 14 retained + 2 agg + 2 lin + 2 pad (16B TMA aligned)
        self.block_q = 16
        self.rows_per_wg = 8
        self.pairs_per_wg = 4
        self.lin_chunk = 8  # cols 128..143
        self.N = 144
        self.num_tmem_stages = 3
        self.num_kv_stages = 2 if bf16 else NUM_KV_STAGES
        self.num_math_threads = 256
        self.threads_per_cta = 384
        self.math_wg_registers = 224
        self.specialized_wg_registers = 56
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (BLOCK_KV, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], k * (HEAD_DIM // k))
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, 2 * self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, b_dtype, 2)
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits) or bf16
        q_pairs: cute.Tensor,  # [wg_blocks * 144, 128, 1] uint8 or bf16
        w_prune: cute.Tensor,  # [num_q, 20] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32
        knsc: cute.Tensor,  # [>= seq_len_kv] f32
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        allowance: cute.Tensor,  # [num_q] f32
        mask: cute.Tensor,  # [2 * tiles * mask_blocks * 4] i32
        flag: cute.Tensor,  # [1] i32
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        expected_flag: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.bf16):
            kv_typed, b = kv, q_pairs
        else:
            kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(q_pairs, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator, cute.make_layout((seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0))
        )
        tiled_mma = self._setup_mma(
            a.element_type,
            b.element_type,
            utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        a_smem_single = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a,
            a_smem_single,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_single = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b,
            b_smem_single,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        w_layout = cute.make_layout((self.block_q, self.w_cols), stride=(self.w_cols, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, w_prune, w_layout, (self.block_q, self.w_cols)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((SPLIT_KV,)), (SPLIT_KV,)
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_load_op, knsc, cute.make_layout((SPLIT_KV,)), (SPLIT_KV,)
        )
        self.num_q_tma_bytes = 2 * cute.size_in_bytes(b.element_type, b_smem_single) + self.block_q * self.w_cols * 4
        self.num_kv_tma_bytes = SPLIT_KV * HEAD_DIM * (a.element_type.width // 8) + 2 * SPLIT_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            raw_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_tmem_stages]
            tmem_holding_buf: Int32

        num_tiles = (num_q + self.block_q - 1) // self.block_q
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
            tma_atom_n,
            tma_tensor_n,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            allowance,
            mask,
            flag,
            num_q,
            seq_len_kv,
            mask_blocks,
            num_splits,
            expected_flag,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
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
        mW: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        mN: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mAllowance: cute.Tensor,
        mMask: cute.Tensor,
        mFlag: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        expected_flag: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        warpgroup_idx = warp_idx // 4
        tile = gdimx - 1 - bidx  # longest windows first
        query_start = tile * self.block_q

        flag_ok = expected_flag == Int32(0)
        if expected_flag != Int32(0):
            flag_ok = mFlag[0] == expected_flag
        total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
        blocks_per_split = (total_blocks + num_splits - 1) // num_splits
        split_lo = bidy * blocks_per_split * BLOCK_KV
        split_hi = min((bidy + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
        range_lo = seq_len_kv
        range_hi = Int32(0)
        cover_lo = Int32(0)
        cover_hi = seq_len_kv
        has_work = cutlass.Boolean(False)
        if flag_ok:
            for r in cutlass.range_constexpr(self.block_q):
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

        if has_work:
            kv_start = range_lo // BLOCK_KV * BLOCK_KV
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
            num_tiles_256 = (num_kv + 1) // 2

            if warp_idx == 8:
                for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s, tma_atom_n):
                    cpasync.prefetch_descriptor(atom)

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_mbar.data_ptr(),
                num_stages=1,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_q_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            kv_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_kv_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            raw_mbar = storage.raw_mbar.data_ptr()
            full_tmem = raw_mbar
            empty_tmem = raw_mbar + self.num_tmem_stages
            if tidx == 0:
                for i in cutlass.range_constexpr(self.num_tmem_stages):
                    cute.arch.mbarrier_init(full_tmem + i, 1)
                    cute.arch.mbarrier_init(empty_tmem + i, 4)
                cute.arch.mbarrier_init_fence()

            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=288),
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
                layout=cute.make_layout(
                    (self.block_q, self.w_cols, 1), stride=(self.w_cols, 1, self.block_q * self.w_cols)
                ),
                byte_alignment=128,
            )
            sVec = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout(
                    (SPLIT_KV, self.num_kv_stages, 2), stride=(1, SPLIT_KV, SPLIT_KV * self.num_kv_stages)
                ),
                byte_alignment=128,
            )
            sPairMeta = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((2, self.pairs_per_wg, 12), stride=(self.pairs_per_wg * 12, 12, 1)),
                byte_alignment=16,
            )
            sWin = smem.allocate_tensor(
                element_type=Int32, layout=cute.make_layout((self.block_q, 2), stride=(2, 1)), byte_alignment=16
            )

            thr_mma = tiled_mma.get_slice(0)
            gA = cute.local_tile(
                cute.domain_offset((kv_start, 0, 0), mA_mkl),
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sKV, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3)
            )
            gS = cute.local_tile(cute.domain_offset((kv_start,), mS), (SPLIT_KV,), (None,))
            gN = cute.local_tile(cute.domain_offset((kv_start,), mN), (SPLIT_KV,), (None,))
            tSsS, tSgS = cpasync.tma_partition(
                tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sVec[(None, None, 0)], 0, 1), gS
            )
            tNsN, tNgN = cpasync.tma_partition(
                tma_atom_n, 0, cute.make_layout(1), cute.group_modes(sVec[(None, None, 1)], 0, 1), gN
            )
            gB = cute.local_tile(
                cute.domain_offset((tile * 2 * self.N, 0, 0), mB_nkl),
                cute.slice_(self.mma_tiler, (0, None, None)),
                (None, None, None),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3)
            )
            gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), (self.block_q, self.w_cols), (None, None))
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(gW, 0, 2)
            )

            tCrA = tiled_mma.make_fragment_A(sKV)
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_tmem_stages))
            epi_sub_mn = (BLOCK_KV, 16)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, (BLOCK_KV, 16), False
            )

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            if warp_idx == 8:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                q_pipeline.producer_acquire(q_prod)
                q_bar = q_pipeline.producer_get_barrier(q_prod)
                cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_b, tBgB[(None, 1, 0, 0)], tBsB[(None, 1)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
                kv_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_kv_stages)
                for t in cutlass.range(num_tiles_256, unroll=1):
                    kv_pipeline.producer_acquire(kv_prod)
                    bar = kv_pipeline.producer_get_barrier(kv_prod)
                    stage = kv_prod.index
                    cute.copy(tma_atom_a, tAgA[(None, 2 * t, 0, 0)], tAsA[(None, 2 * stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_a, tAgA[(None, 2 * t + 1, 0, 0)], tAsA[(None, 2 * stage + 1)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_s, tSgS[(None, t)], tSsS[(None, stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_n, tNgN[(None, t)], tNsN[(None, stage)], tma_bar_ptr=bar)
                    kv_prod.advance()
            elif warp_idx == 9:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                named_barrier_sync_unaligned(288)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
                num_k = cute.size(tCrA.shape[2])
                q_pipeline.consumer_wait(q_cons)
                step = Int32(0)
                nstages = self.num_tmem_stages
                for _t in cutlass.range(num_tiles_256, unroll=1):
                    ready = kv_pipeline.consumer_try_wait(kv_cons)
                    kv_pipeline.consumer_wait(kv_cons, ready)
                    kv_stage = kv_cons.index
                    for half in cutlass.range_constexpr(2):
                        a_slice = 2 * kv_stage + half
                        for g in cutlass.range_constexpr(2):
                            ts = step % nstages
                            tph = (step // nstages) % 2
                            cute.arch.mbarrier_wait(empty_tmem + ts, tph ^ 1)
                            acc = tCtAcc_base[(None, None, None, ts)]
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for k in cutlass.range_constexpr(num_k):
                                cute.gemm(
                                    tiled_mma, acc, tCrA[None, None, k, a_slice], tCrB[None, None, k, g], acc
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                tcgen05.commit(full_tmem + ts)
                            step = step + 1
                    kv_cons.advance()
            elif warp_idx < 8:
                self._run_math_warp(
                    num_tiles_256,
                    kv_start,
                    num_kv,
                    cover_lo,
                    cover_hi,
                    split_lo,
                    split_hi,
                    query_start,
                    tile,
                    num_q,
                    seq_len_kv,
                    mask_blocks,
                    copy_atom_t2r,
                    epi_sub_mn,
                    kv_pipeline,
                    q_pipeline,
                    full_tmem,
                    empty_tmem,
                    sVec,
                    sW,
                    sPairMeta,
                    sWin,
                    tCtAcc_fake,
                    tidx,
                    tmem,
                    warpgroup_idx,
                    mRowStart,
                    mRowEnd,
                    mOrigin,
                    mInvDelta,
                    mThBucket,
                    mAllowance,
                    mMask,
                )
            else:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self,
        num_tiles_256,
        kv_start,
        num_kv,
        cover_lo,
        cover_hi,
        split_lo,
        split_hi,
        query_start,
        tile,
        num_q,
        seq_len_kv,
        mask_blocks,
        copy_atom_t2r,
        epi_sub_mn,
        kv_pipeline,
        q_pipeline,
        full_tmem,
        empty_tmem,
        sVec,
        sW,
        sPairMeta,
        sWin,
        tCtAcc_fake,
        tidx,
        tmem,
        warpgroup_idx,
        mRowStart,
        mRowEnd,
        mOrigin,
        mInvDelta,
        mThBucket,
        mAllowance,
        mMask,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        lane = cute.arch.lane_idx()
        quad = cute.arch.make_warp_uniform(cute.arch.warp_idx() % 4)
        tmem.allocate(TMEM_TOTAL_COLS)
        named_barrier_sync_unaligned(288)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        epi_tidx = tidx % 128

        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
        cC = cute.make_identity_tensor(epi_sub_mn)
        tTR_cC = thr_copy_t2r.partition_D(cC)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        tTR_rAcc2 = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        m_coord = tTR_cC[0][0]

        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        q_pipeline.consumer_wait(q_cons)
        R = self.retained
        if tidx < self.block_q:
            r = tidx
            g_r = r // self.rows_per_wg
            lr = r % self.rows_per_wg
            p_r = lr // 2
            pr = lr % 2
            row_q = query_start + r
            row = min(row_q, num_q - 1)
            sPairMeta[(g_r, p_r, 0 + pr)] = sW[(r, R, 0)]
            sPairMeta[(g_r, p_r, 2 + pr)] = sW[(r, R + 1, 0)]
            sPairMeta[(g_r, p_r, 4 + pr)] = sW[(r, R + 2, 0)]
            sPairMeta[(g_r, p_r, 6 + pr)] = sW[(r, R + 3, 0)]
            sPairMeta[(g_r, p_r, 8 + pr)] = mAllowance[row]
            sPairMeta[(g_r, p_r, 10 + pr)] = loose_raw_gate(mInvDelta[row], mOrigin[row], mThBucket[row])
            s = seq_len_kv
            e = Int32(0)
            if row_q < num_q:
                s = max(min(mRowStart[row_q], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row_q], seq_len_kv), split_hi)
            sWin[(r, 0)] = s
            sWin[(r, 1)] = e
        cute.arch.sync_warp()
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()

        # Keep all 4 x 28 = 112 halved weights and 4 x 12 = 48 pair metadata floats in registers.
        row0 = warpgroup_idx * self.rows_per_wg
        w_cache = cute.make_rmem_tensor(cute.make_layout((self.pairs_per_wg, 2 * R), stride=(2 * R, 1)), Float32)
        pm = cute.make_rmem_tensor(cute.make_layout((self.pairs_per_wg, 12), stride=(12, 1)), Float32)
        for p in cutlass.range_constexpr(self.pairs_per_wg):
            for c in cutlass.range_constexpr(R):
                w_cache[(p, 2 * c)] = sW[(row0 + 2 * p, c, 0)] * Float32(0.5)
                w_cache[(p, 2 * c + 1)] = sW[(row0 + 2 * p + 1, c, 0)] * Float32(0.5)
            for m_i in cutlass.range_constexpr(12):
                pm[(p, m_i)] = sPairMeta[(warpgroup_idx, p, m_i)]

        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
        nstages = self.num_tmem_stages
        step = warpgroup_idx
        tile_base = ((2 * tile + warpgroup_idx) * mask_blocks) * 4 + quad

        for t in cutlass.range(num_tiles_256, unroll=1):
            kv_stage = kv_cons.index
            for half in cutlass.range_constexpr(2):
                j = 2 * t + half
                ts = step % nstages
                tph = (step // nstages) % 2
                cute.arch.mbarrier_wait(full_tmem + ts, tph)
                cute.arch.sync_warp()

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, ts)]

                # Double-buffered TMEM readout: hardware register scoreboard tracks per-buffer readiness;
                # only one fence_view_async_tmem_load is needed right before releasing the TMEM slot.
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, self.lin_chunk)], tTR_rAcc)
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc2)
                scale = sVec[(half * BLOCK_KV + m_coord, kv_stage, 0)]
                norm = sVec[(half * BLOCK_KV + m_coord, kv_stage, 1)]
                if cutlass.const_expr(half == 1):
                    cute.arch.sync_warp()
                    kv_pipeline.consumer_release(kv_cons)
                    kv_cons.advance()
                lin_vec = tTR_rAcc.load()

                blk_lo = kv_start + j * BLOCK_KV
                kv_idx = blk_lo + m_coord
                interior = (blk_lo >= cover_lo) & (blk_lo + BLOCK_KV <= cover_hi)
                surv = cutlass.Boolean(False)

                for p in cutlass.range_constexpr(self.pairs_per_wg):
                    acc0 = cute.arch.mul_packed_f32x2(
                        (lin_vec[4 * p], lin_vec[4 * p + 1]), (pm[(p, 4)], pm[(p, 5)]), rnd=_RND_RN
                    )
                    acc1 = cute.arch.mul_packed_f32x2(
                        (lin_vec[4 * p + 2], lin_vec[4 * p + 3]), (pm[(p, 6)], pm[(p, 7)]), rnd=_RND_RN
                    )

                    # Chunk 2*p: retained heads 0..7 (16 floats = 8 heads x 2 rows).
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 2 * p + 1)], tTR_rAcc)
                    v0 = tTR_rAcc2.load()
                    for c in cutlass.range_constexpr(0, 8, 2):
                        acc0 = cute.arch.fma_packed_f32x2(
                            (abs_f32(v0[2 * c]), abs_f32(v0[2 * c + 1])),
                            (w_cache[(p, 2 * c)], w_cache[(p, 2 * c + 1)]),
                            acc0,
                            rnd=_RND_RN,
                        )
                        acc1 = cute.arch.fma_packed_f32x2(
                            (abs_f32(v0[2 * c + 2]), abs_f32(v0[2 * c + 3])),
                            (w_cache[(p, 2 * c + 2)], w_cache[(p, 2 * c + 3)]),
                            acc1,
                            rnd=_RND_RN,
                        )

                    # Chunk 2*p + 1: retained heads 8..13 (12 floats) + agg_hi/agg_lo (4 floats).
                    if cutlass.const_expr(p + 1 < self.pairs_per_wg):
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 2 * p + 2)], tTR_rAcc2)
                    else:
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(empty_tmem + ts)
                    v1 = tTR_rAcc.load()
                    for c in cutlass.range_constexpr(0, 6, 2):
                        w_off = 16 + 2 * c
                        acc0 = cute.arch.fma_packed_f32x2(
                            (abs_f32(v1[2 * c]), abs_f32(v1[2 * c + 1])),
                            (w_cache[(p, w_off)], w_cache[(p, w_off + 1)]),
                            acc0,
                            rnd=_RND_RN,
                        )
                        acc1 = cute.arch.fma_packed_f32x2(
                            (abs_f32(v1[2 * c + 2]), abs_f32(v1[2 * c + 3])),
                            (w_cache[(p, w_off + 2)], w_cache[(p, w_off + 3)]),
                            acc1,
                            rnd=_RND_RN,
                        )
                    agg_hi = (v1[12], v1[13])
                    agg_lo = (v1[14], v1[15])

                    agg = cute.arch.fma_packed_f32x2(
                        agg_lo,
                        (pm[(p, 2)], pm[(p, 3)]),
                        cute.arch.mul_packed_f32x2(agg_hi, (pm[(p, 0)], pm[(p, 1)]), rnd=_RND_RN),
                        rnd=_RND_RN,
                    )
                    reduced = cute.arch.add_packed_f32x2(
                        cute.arch.add_packed_f32x2(acc0, acc1, rnd=_RND_RN),
                        (-fmax(agg[0], Float32(0.0)), -fmax(agg[1], Float32(0.0))),
                        rnd=_RND_RN,
                    )
                    bound = cute.arch.fma_packed_f32x2(
                        (pm[(p, 8)], pm[(p, 9)]),
                        (norm, norm),
                        cute.arch.mul_packed_f32x2((scale, scale), reduced, rnd=_RND_RN),
                        rnd=_RND_RN,
                    )
                    g0 = pm[(p, 10)]
                    g1 = pm[(p, 11)]
                    keep0 = (bound[0] > g0) | (bound[0] != bound[0]) | (g0 != g0)
                    keep1 = (bound[1] > g1) | (bound[1] != bound[1]) | (g1 != g1)
                    r0 = row0 + 2 * p
                    r1 = r0 + 1
                    in0 = interior | ((kv_idx >= sWin[(r0, 0)]) & (kv_idx < sWin[(r0, 1)]))
                    in1 = interior | ((kv_idx >= sWin[(r1, 0)]) & (kv_idx < sWin[(r1, 1)]))
                    surv = surv | (keep0 & in0) | (keep1 & in1)

                word = Int32(cute.arch.vote_ballot_sync(surv))
                if (lane == 0) & (j < num_kv):
                    gblock = kv_start // BLOCK_KV + j
                    base = mMask.iterator.toint() + cutlass.Int64(tile_base + gblock * 4) * 4
                    store_global_b32(base, word)
                step = step + 2

        q_pipeline.consumer_release(q_cons)
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


class Prune24Sm100:
    """24-col SM100 score-upper-bound pruning kernel (R=22 retained heads, 16 rows/CTA).

    Row-split warpgroups over shared 128-key KV tiles (halving KV HBM traffic vs 8-row tiles):
      * FP8 (N=208): lossless linear-term split (22 retained + 2 agg + 2 lin cols, 0 ReLU ops),
        8 x 128-key shared KV stages (~197 KB SMEM), 2 x 256 = 512 TMEM cols;
      * BF16 (N=192): stays at 24 cols (BF16 Tensor Cores are FLOPS-bound), 3 x 128-key shared KV
        stages (~204 KB SMEM), 2 x 256 = 512 TMEM cols;
      * Double-buffered 16-col TMEM readout with a single fence at release, 112 register-cached
        retained weights (0 stack spill), and early KV stage release;
      * Warp roles (384 threads): warps 0-7 math, warp 8 TMA, warp 9 UMMA, warps 10-11 unused.
    """

    def __init__(self, *, bf16: bool = False):
        self.bf16 = bf16
        self.use_lin = not bf16
        self.cols = COLS_24
        self.retained = COLS_24 - 2  # 22
        self.q_cols = COLS_24 + 2 if self.use_lin else COLS_24  # 26 FP8, 24 BF16
        self.w_cols = 28 if self.use_lin else 24  # 16B TMA aligned
        self.block_q = 16
        self.rows_per_wg = 8
        self.pairs_per_wg = 4
        self.lin_chunk = 12  # cols 192..207 (FP8 only)
        self.pair_floats = 2 * self.q_cols + 4  # 56 FP8, 52 BF16
        self.num_kv_stages = 3 if bf16 else NUM_KV_STAGES_24
        self.num_w_reg = NUM_W_REG_24 if self.use_lin else 16  # 0 stack spill for FP8; BF16 uses key-split Prune24Bf16Sm100
        self.N = self.rows_per_wg * self.cols + (16 if self.use_lin else 0)  # 208 FP8, 192 BF16
        self.num_math_threads = 256
        self.threads_per_cta = 384
        # 256 * 224 + 128 * 56 = 64,512 == 384 * 168 regs: 0 stack spills in math & TMA/UMMA warps.
        self.math_wg_registers = 224
        self.specialized_wg_registers = 56
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (BLOCK_KV, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], k * (HEAD_DIM // k))
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, b_dtype, 2)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tiled_mma.make_fragment_C(acc_shape))
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,
        q_pairs: cute.Tensor,
        w_prune: cute.Tensor,
        kv_scales: cute.Tensor,
        knsc: cute.Tensor,
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        allowance: cute.Tensor,
        mask: cute.Tensor,
        flag: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        expected_flag: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.bf16):
            kv_typed, b = kv, q_pairs
        else:
            kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(q_pairs, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator, cute.make_layout((seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0))
        )
        tiled_mma = self._setup_mma(
            a.element_type,
            b.element_type,
            utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        a_smem_single = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a,
            a_smem_single,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_single = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b,
            b_smem_single,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        w_layout = cute.make_layout((self.block_q, self.w_cols), stride=(self.w_cols, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, w_prune, w_layout, (self.block_q, self.w_cols)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_load_op, knsc, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        self.num_q_tma_bytes = 2 * cute.size_in_bytes(b.element_type, b_smem_single) + self.block_q * self.w_cols * 4
        self.num_kv_tma_bytes = BLOCK_KV * HEAD_DIM * (a.element_type.width // 8) + 2 * BLOCK_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding_buf: Int32

        num_tiles = (num_q + self.block_q - 1) // self.block_q
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
            tma_atom_n,
            tma_tensor_n,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            allowance,
            mask,
            flag,
            num_q,
            seq_len_kv,
            mask_blocks,
            num_splits,
            expected_flag,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
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
        mW: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        mN: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mAllowance: cute.Tensor,
        mMask: cute.Tensor,
        mFlag: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        expected_flag: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        warpgroup_idx = warp_idx // 4
        tile = gdimx - 1 - bidx  # longest windows first
        query_start = tile * self.block_q

        flag_ok = expected_flag == Int32(0)
        if expected_flag != Int32(0):
            flag_ok = mFlag[0] == expected_flag
        total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
        blocks_per_split = (total_blocks + num_splits - 1) // num_splits
        split_lo = bidy * blocks_per_split * BLOCK_KV
        split_hi = min((bidy + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
        range_lo = seq_len_kv
        range_hi = Int32(0)
        cover_lo = Int32(0)
        cover_hi = seq_len_kv
        has_work = cutlass.Boolean(False)
        if flag_ok:
            for r in cutlass.range_constexpr(self.block_q):
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

        if has_work:
            kv_start = range_lo // BLOCK_KV * BLOCK_KV
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV

            if warp_idx == 8:
                for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s, tma_atom_n):
                    cpasync.prefetch_descriptor(atom)

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_mbar.data_ptr(),
                num_stages=1,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_q_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            kv_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_kv_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            math_ring = min(warpgroup_idx, Int32(1))
            umma_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
            umma_pipes = [
                pipeline.PipelineUmmaAsync.create(
                    barrier_storage=storage.umma_mbar.data_ptr() + g * 2,
                    num_stages=1,
                    producer_group=prod_group,
                    consumer_group=umma_cons_group,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
                for g in range(2)
            ]
            umma_pipeline_math = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar.data_ptr() + math_ring * 2,
                num_stages=1,
                producer_group=prod_group,
                consumer_group=umma_cons_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=288),
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
                layout=cute.make_layout(
                    (self.block_q, self.w_cols, 1), stride=(self.w_cols, 1, self.block_q * self.w_cols)
                ),
                byte_alignment=128,
            )
            sVec = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout(
                    (BLOCK_KV, self.num_kv_stages, 2), stride=(1, BLOCK_KV, BLOCK_KV * self.num_kv_stages)
                ),
                byte_alignment=128,
            )
            sPair = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout(
                    (2, self.pairs_per_wg, self.pair_floats),
                    stride=(self.pairs_per_wg * self.pair_floats, self.pair_floats, 1),
                ),
                byte_alignment=16,
            )
            sWin = smem.allocate_tensor(
                element_type=Int32, layout=cute.make_layout((self.block_q, 2), stride=(2, 1)), byte_alignment=16
            )

            thr_mma = tiled_mma.get_slice(0)
            gA = cute.local_tile(
                cute.domain_offset((kv_start, 0, 0), mA_mkl),
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sKV, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3)
            )
            gS = cute.local_tile(cute.domain_offset((kv_start,), mS), (BLOCK_KV,), (None,))
            gN = cute.local_tile(cute.domain_offset((kv_start,), mN), (BLOCK_KV,), (None,))
            tSsS, tSgS = cpasync.tma_partition(
                tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sVec[(None, None, 0)], 0, 1), gS
            )
            tNsN, tNgN = cpasync.tma_partition(
                tma_atom_n, 0, cute.make_layout(1), cute.group_modes(sVec[(None, None, 1)], 0, 1), gN
            )
            gB = cute.local_tile(
                cute.domain_offset((tile * 2 * self.N, 0, 0), mB_nkl),
                cute.slice_(self.mma_tiler, (0, None, None)),
                (None, None, None),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3)
            )
            gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), (self.block_q, self.w_cols), (None, None))
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(gW, 0, 2)
            )

            tCrA = tiled_mma.make_fragment_A(sKV)
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
            epi_sub_mn = (BLOCK_KV, 16)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype,
                (BLOCK_KV, 16), False,
            )

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            if warp_idx == 8:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                q_pipeline.producer_acquire(q_prod)
                q_bar = q_pipeline.producer_get_barrier(q_prod)
                cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_b, tBgB[(None, 1, 0, 0)], tBsB[(None, 1)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
                kv_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_kv_stages)
                for j in cutlass.range(num_kv, unroll=1):
                    kv_pipeline.producer_acquire(kv_prod)
                    bar = kv_pipeline.producer_get_barrier(kv_prod)
                    stage = kv_prod.index
                    cute.copy(tma_atom_a, tAgA[(None, j, 0, 0)], tAsA[(None, stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_s, tSgS[(None, j)], tSsS[(None, stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_n, tNgN[(None, j)], tNsN[(None, stage)], tma_bar_ptr=bar)
                    kv_prod.advance()
            elif warp_idx == 9:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                named_barrier_sync_unaligned(288)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                acc_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[(None, None, None, 0)]
                acc_1 = cute.make_tensor(tmem_ptr + self.num_tmem_alloc_cols, tCtAcc_fake.layout)[(None, None, None, 0)]
                q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
                umma_prod_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                umma_prod_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                num_k = cute.size(tCrA.shape[2])
                q_pipeline.consumer_wait(q_cons)
                for _j in cutlass.range(num_kv, unroll=1):
                    ready = kv_pipeline.consumer_try_wait(kv_cons)
                    kv_pipeline.consumer_wait(kv_cons, ready)
                    kv_stage = kv_cons.index

                    empty_0 = umma_pipes[0].producer_try_acquire(umma_prod_0)
                    umma_pipes[0].producer_acquire(umma_prod_0, empty_0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(num_k):
                        cute.gemm(tiled_mma, acc_0, tCrA[None, None, k, kv_stage], tCrB[None, None, k, 0], acc_0)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    umma_pipes[0].producer_commit(umma_prod_0)
                    umma_prod_0.advance()

                    empty_1 = umma_pipes[1].producer_try_acquire(umma_prod_1)
                    umma_pipes[1].producer_acquire(umma_prod_1, empty_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(num_k):
                        cute.gemm(tiled_mma, acc_1, tCrA[None, None, k, kv_stage], tCrB[None, None, k, 1], acc_1)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    umma_pipes[1].producer_commit(umma_prod_1)
                    umma_prod_1.advance()

                    kv_cons.advance()
            elif warp_idx < 8:
                self._run_math_warp(
                    kv_start,
                    num_kv,
                    cover_lo,
                    cover_hi,
                    split_lo,
                    split_hi,
                    query_start,
                    tile,
                    num_q,
                    seq_len_kv,
                    mask_blocks,
                    copy_atom_t2r,
                    epi_sub_mn,
                    kv_pipeline,
                    q_pipeline,
                    sVec,
                    sW,
                    sPair,
                    sWin,
                    tCtAcc_fake,
                    tidx,
                    tmem,
                    umma_pipeline_math,
                    warpgroup_idx,
                    mRowStart,
                    mRowEnd,
                    mOrigin,
                    mInvDelta,
                    mThBucket,
                    mAllowance,
                    mMask,
                )
            else:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self,
        kv_start,
        num_kv,
        cover_lo,
        cover_hi,
        split_lo,
        split_hi,
        query_start,
        tile,
        num_q,
        seq_len_kv,
        mask_blocks,
        copy_atom_t2r,
        epi_sub_mn,
        kv_pipeline,
        q_pipeline,
        sVec,
        sW,
        sPair,
        sWin,
        tCtAcc_fake,
        tidx,
        tmem,
        umma_pipeline_math,
        warpgroup_idx,
        mRowStart,
        mRowEnd,
        mOrigin,
        mInvDelta,
        mThBucket,
        mAllowance,
        mMask,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        lane = cute.arch.lane_idx()
        quad = cute.arch.make_warp_uniform(cute.arch.warp_idx() % 4)
        tmem.allocate(2 * self.num_tmem_alloc_cols)
        named_barrier_sync_unaligned(288)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr + warpgroup_idx * self.num_tmem_alloc_cols, tCtAcc_fake.layout)
        epi_tidx = tidx % 128
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, None)], epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)[(None, None, None, None, None, 0)]
        cC = cute.make_identity_tensor(epi_sub_mn)
        tTR_cC = thr_copy_t2r.partition_D(cC)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        tTR_rAcc2 = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        m_coord = tTR_cC[0][0]

        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        q_pipeline.consumer_wait(q_cons)
        q_cols = self.q_cols
        allow_off = 2 * q_cols
        gate_off = 2 * q_cols + 2
        total_w = self.block_q * q_cols
        for step_w in cutlass.range_constexpr(2):
            pos = tidx + step_w * self.num_math_threads
            if pos < total_w:
                r = pos // q_cols
                c = pos % q_cols
                w = sW[(r, c, 0)]
                if c < self.retained:
                    w = w * Float32(0.5)
                g_r = r // self.rows_per_wg
                lr = r % self.rows_per_wg
                sPair[(g_r, lr // 2, 2 * c + lr % 2)] = w
        if tidx < self.block_q:
            r = tidx
            g_r = r // self.rows_per_wg
            lr = r % self.rows_per_wg
            row_q = query_start + r
            row = min(row_q, num_q - 1)
            sPair[(g_r, lr // 2, allow_off + lr % 2)] = mAllowance[row]
            sPair[(g_r, lr // 2, gate_off + lr % 2)] = loose_raw_gate(
                mInvDelta[row], mOrigin[row], mThBucket[row]
            )
            s = seq_len_kv
            e = Int32(0)
            if row_q < num_q:
                s = max(min(mRowStart[row_q], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row_q], seq_len_kv), split_hi)
            sWin[(r, 0)] = s
            sWin[(r, 1)] = e
        cute.arch.sync_warp()
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()

        # Cache the first num_w_reg/2 retained heads' halved pair weights in registers (0 stack spill).
        num_w_reg = self.num_w_reg
        w_reg = cute.make_rmem_tensor(cute.make_layout((self.pairs_per_wg, num_w_reg), stride=(num_w_reg, 1)), Float32)
        for pp in cutlass.range_constexpr(self.pairs_per_wg):
            for wi in cutlass.range_constexpr(num_w_reg):
                w_reg[(pp, wi)] = sPair[(warpgroup_idx, pp, wi)]

        row0 = warpgroup_idx * self.rows_per_wg
        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
        umma_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        tile_base = ((2 * tile + warpgroup_idx) * mask_blocks) * 4 + quad
        for j in cutlass.range(num_kv, unroll=1):
            umma_pipeline_math.consumer_wait(umma_cons)
            cute.arch.sync_warp()
            # Double-buffered 16-col TMEM readout: prefetch chunk 12 (FP8 linear terms) and chunk 0.
            if cutlass.const_expr(self.use_lin):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, self.lin_chunk)], tTR_rAcc)
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc2)
            scale = sVec[(m_coord, kv_cons.index, 0)]
            norm = sVec[(m_coord, kv_cons.index, 1)]
            # Both WGs arrive on kv_pipeline (consumer_group=8) after their UMMA completes and scale/norm are in regs.
            cute.arch.sync_warp()
            kv_pipeline.consumer_release(kv_cons)
            kv_cons.advance()
            lin_vec = tTR_rAcc.load() if cutlass.const_expr(self.use_lin) else None

            blk_lo = kv_start + j * BLOCK_KV
            kv_idx = blk_lo + m_coord
            interior = (blk_lo >= cover_lo) & (blk_lo + BLOCK_KV <= cover_hi)
            surv = cutlass.Boolean(False)

            for p in cutlass.range_constexpr(self.pairs_per_wg):
                if cutlass.const_expr(p % 2 == 0):
                    buf_a, buf_b = tTR_rAcc2, tTR_rAcc
                else:
                    buf_a, buf_b = tTR_rAcc, tTR_rAcc2

                if cutlass.const_expr(self.use_lin):
                    acc0 = cute.arch.mul_packed_f32x2(
                        (lin_vec[4 * p], lin_vec[4 * p + 1]),
                        (sPair[(warpgroup_idx, p, 48)], sPair[(warpgroup_idx, p, 49)]),
                        rnd=_RND_RN,
                    )
                    acc1 = cute.arch.mul_packed_f32x2(
                        (lin_vec[4 * p + 2], lin_vec[4 * p + 3]),
                        (sPair[(warpgroup_idx, p, 50)], sPair[(warpgroup_idx, p, 51)]),
                        rnd=_RND_RN,
                    )
                else:
                    acc0 = (Float32(0.0), Float32(0.0))
                    acc1 = (Float32(0.0), Float32(0.0))

                # Chunk 3*p: retained heads 0..7 (16 floats = 8 heads x 2 rows).
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 3 * p + 1)], buf_b)
                v0 = buf_a.load()
                for c in cutlass.range_constexpr(0, 8, 2):
                    if cutlass.const_expr(self.use_lin):
                        t0 = (abs_f32(v0[2 * c]), abs_f32(v0[2 * c + 1]))
                        t1 = (abs_f32(v0[2 * c + 2]), abs_f32(v0[2 * c + 3]))
                    else:
                        t0 = cute.arch.add_packed_f32x2(
                            (v0[2 * c], v0[2 * c + 1]), (abs_f32(v0[2 * c]), abs_f32(v0[2 * c + 1])), rnd=_RND_RN
                        )
                        t1 = cute.arch.add_packed_f32x2(
                            (v0[2 * c + 2], v0[2 * c + 3]), (abs_f32(v0[2 * c + 2]), abs_f32(v0[2 * c + 3])), rnd=_RND_RN
                        )
                    acc0 = cute.arch.fma_packed_f32x2(
                        t0, (w_reg[(p, 2 * c)], w_reg[(p, 2 * c + 1)]), acc0, rnd=_RND_RN
                    )
                    acc1 = cute.arch.fma_packed_f32x2(
                        t1, (w_reg[(p, 2 * c + 2)], w_reg[(p, 2 * c + 3)]), acc1, rnd=_RND_RN
                    )

                # Chunk 3*p + 1: retained heads 8..15 (16 floats = 8 heads x 2 rows).
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 3 * p + 2)], buf_a)
                v1 = buf_b.load()
                for c in cutlass.range_constexpr(0, 8, 2):
                    w_off = 16 + 2 * c
                    w00 = w_reg[(p, w_off)] if cutlass.const_expr(w_off < num_w_reg) else sPair[(warpgroup_idx, p, w_off)]
                    w01 = w_reg[(p, w_off + 1)] if cutlass.const_expr(w_off + 1 < num_w_reg) else sPair[(warpgroup_idx, p, w_off + 1)]
                    w10 = w_reg[(p, w_off + 2)] if cutlass.const_expr(w_off + 2 < num_w_reg) else sPair[(warpgroup_idx, p, w_off + 2)]
                    w11 = w_reg[(p, w_off + 3)] if cutlass.const_expr(w_off + 3 < num_w_reg) else sPair[(warpgroup_idx, p, w_off + 3)]
                    if cutlass.const_expr(self.use_lin):
                        t0 = (abs_f32(v1[2 * c]), abs_f32(v1[2 * c + 1]))
                        t1 = (abs_f32(v1[2 * c + 2]), abs_f32(v1[2 * c + 3]))
                    else:
                        t0 = cute.arch.add_packed_f32x2(
                            (v1[2 * c], v1[2 * c + 1]), (abs_f32(v1[2 * c]), abs_f32(v1[2 * c + 1])), rnd=_RND_RN
                        )
                        t1 = cute.arch.add_packed_f32x2(
                            (v1[2 * c + 2], v1[2 * c + 3]), (abs_f32(v1[2 * c + 2]), abs_f32(v1[2 * c + 3])), rnd=_RND_RN
                        )
                    acc0 = cute.arch.fma_packed_f32x2(t0, (w00, w01), acc0, rnd=_RND_RN)
                    acc1 = cute.arch.fma_packed_f32x2(t1, (w10, w11), acc1, rnd=_RND_RN)

                # Chunk 3*p + 2: retained heads 16..21 (12 floats) + agg_hi/agg_lo (4 floats).
                if cutlass.const_expr(p + 1 < self.pairs_per_wg):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 3 * p + 3)], buf_b)
                else:
                    cute.arch.fence_view_async_tmem_load()
                    umma_pipeline_math.consumer_release(umma_cons)
                    umma_cons.advance()
                v2 = buf_a.load()
                for c in cutlass.range_constexpr(0, 6, 2):
                    w_off = 32 + 2 * c
                    if cutlass.const_expr(self.use_lin):
                        t0 = (abs_f32(v2[2 * c]), abs_f32(v2[2 * c + 1]))
                        t1 = (abs_f32(v2[2 * c + 2]), abs_f32(v2[2 * c + 3]))
                    else:
                        t0 = cute.arch.add_packed_f32x2(
                            (v2[2 * c], v2[2 * c + 1]), (abs_f32(v2[2 * c]), abs_f32(v2[2 * c + 1])), rnd=_RND_RN
                        )
                        t1 = cute.arch.add_packed_f32x2(
                            (v2[2 * c + 2], v2[2 * c + 3]), (abs_f32(v2[2 * c + 2]), abs_f32(v2[2 * c + 3])), rnd=_RND_RN
                        )
                    acc0 = cute.arch.fma_packed_f32x2(
                        t0, (sPair[(warpgroup_idx, p, w_off)], sPair[(warpgroup_idx, p, w_off + 1)]), acc0, rnd=_RND_RN
                    )
                    acc1 = cute.arch.fma_packed_f32x2(
                        t1, (sPair[(warpgroup_idx, p, w_off + 2)], sPair[(warpgroup_idx, p, w_off + 3)]), acc1, rnd=_RND_RN
                    )
                agg = cute.arch.fma_packed_f32x2(
                    (v2[14], v2[15]),
                    (sPair[(warpgroup_idx, p, 46)], sPair[(warpgroup_idx, p, 47)]),
                    cute.arch.mul_packed_f32x2(
                        (v2[12], v2[13]),
                        (sPair[(warpgroup_idx, p, 44)], sPair[(warpgroup_idx, p, 45)]),
                        rnd=_RND_RN,
                    ),
                    rnd=_RND_RN,
                )
                reduced = cute.arch.add_packed_f32x2(
                    cute.arch.add_packed_f32x2(acc0, acc1, rnd=_RND_RN),
                    (-fmax(agg[0], Float32(0.0)), -fmax(agg[1], Float32(0.0))),
                    rnd=_RND_RN,
                )
                bound = cute.arch.fma_packed_f32x2(
                    (sPair[(warpgroup_idx, p, allow_off)], sPair[(warpgroup_idx, p, allow_off + 1)]),
                    (norm, norm),
                    cute.arch.mul_packed_f32x2((scale, scale), reduced, rnd=_RND_RN),
                    rnd=_RND_RN,
                )
                g0 = sPair[(warpgroup_idx, p, gate_off)]
                g1 = sPair[(warpgroup_idx, p, gate_off + 1)]
                keep0 = (bound[0] > g0) | (bound[0] != bound[0]) | (g0 != g0)
                keep1 = (bound[1] > g1) | (bound[1] != bound[1]) | (g1 != g1)
                r0 = row0 + 2 * p
                r1 = r0 + 1
                in0 = interior | ((kv_idx >= sWin[(r0, 0)]) & (kv_idx < sWin[(r0, 1)]))
                in1 = interior | ((kv_idx >= sWin[(r1, 0)]) & (kv_idx < sWin[(r1, 1)]))
                surv = surv | (keep0 & in0) | (keep1 & in1)

            word = cute.arch.vote_ballot_sync(surv)
            if lane == 0:
                gblock = kv_start // BLOCK_KV + j
                base = mMask.iterator.toint() + cutlass.Int64(tile_base + gblock * 4) * 4
                store_global_b32(base, word)
            cute.arch.sync_warp()

        q_pipeline.consumer_release(q_cons)
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


class Prune24Bf16Sm100:
    """24-col SM100 score-upper-bound pruning kernel (R=22 retained heads, 8 rows/CTA, N=192).

    Uses key-split warpgroups (one M=128 x N=192 UMMA per 128-key block, zero cross-WG mask
    reduction) with the first 16 retained heads' weights (128 floats) cached in registers and
    early KV stage release right after reading per-key scale and norm.
    """

    def __init__(self, *, bf16: bool = False):
        self.bf16 = bf16
        self.cols = COLS_24
        self.block_q = 8
        self.pairs_per_subtile = 1
        self.pair_floats = 2 * COLS_24 + 4  # 52
        self.num_kv_stages = 2 if bf16 else 4
        self.N = self.block_q * self.cols  # 192
        self.num_math_threads = 256
        self.threads_per_cta = 384
        self.math_wg_registers = 224
        self.specialized_wg_registers = 56
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (BLOCK_KV, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], k * (HEAD_DIM // k))
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, b_dtype, 1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tiled_mma.make_fragment_C(acc_shape))
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,
        q_pairs: cute.Tensor,
        w_prune: cute.Tensor,
        kv_scales: cute.Tensor,
        knsc: cute.Tensor,
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        allowance: cute.Tensor,
        mask: cute.Tensor,
        flag: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        expected_flag: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.bf16):
            kv_typed, b = kv, q_pairs
        else:
            kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(q_pairs, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator, cute.make_layout((seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0))
        )
        tiled_mma = self._setup_mma(
            a.element_type,
            b.element_type,
            utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a,
            self.a_smem_layout_staged,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        w_layout = cute.make_layout((self.block_q, self.cols), stride=(self.cols, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, w_prune, w_layout, (self.block_q, self.cols)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_load_op, knsc, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        self.num_q_tma_bytes = cute.size_in_bytes(b.element_type, b_smem_layout) + self.block_q * self.cols * 4
        self.num_kv_tma_bytes = BLOCK_KV * HEAD_DIM * (a.element_type.width // 8) + 2 * BLOCK_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 4]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding_buf: Int32

        num_tiles = (num_q + self.block_q - 1) // self.block_q
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
            tma_atom_n,
            tma_tensor_n,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            allowance,
            mask,
            flag,
            num_q,
            seq_len_kv,
            mask_blocks,
            num_splits,
            expected_flag,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
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
        mW: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        mN: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mAllowance: cute.Tensor,
        mMask: cute.Tensor,
        mFlag: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        expected_flag: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        warpgroup_idx = warp_idx // 4
        tile = gdimx - 1 - bidx
        query_start = tile * self.block_q

        flag_ok = expected_flag == Int32(0)
        if expected_flag != Int32(0):
            flag_ok = mFlag[0] == expected_flag
        total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
        blocks_per_split = (total_blocks + num_splits - 1) // num_splits
        split_lo = bidy * blocks_per_split * BLOCK_KV
        split_hi = min((bidy + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
        range_lo = seq_len_kv
        range_hi = Int32(0)
        cover_lo = Int32(0)
        cover_hi = seq_len_kv
        has_work = cutlass.Boolean(False)
        if flag_ok:
            for r in cutlass.range_constexpr(self.block_q):
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

        if has_work:
            kv_start = range_lo // BLOCK_KV * BLOCK_KV
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
            num_pairs = (num_kv + NUM_MATH_WG - 1) // NUM_MATH_WG

            if (warp_idx == 8) | (warp_idx == 9):
                for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s, tma_atom_n):
                    cpasync.prefetch_descriptor(atom)

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_mbar.data_ptr(),
                num_stages=1,
                producer_group=prod_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
                tx_count=self.num_q_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
            kv_pipes = [
                pipeline.PipelineTmaAsync.create(
                    barrier_storage=storage.kv_mbar.data_ptr() + g * self.num_kv_stages * 2,
                    num_stages=self.num_kv_stages,
                    producer_group=prod_group,
                    consumer_group=kv_cons_group,
                    tx_count=self.num_kv_tma_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    tidx=tidx,
                    defer_sync=True,
                )
                for g in range(2)
            ]
            math_ring = min(warpgroup_idx, Int32(1))
            kv_pipeline_math = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar.data_ptr() + math_ring * self.num_kv_stages * 2,
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            umma_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
            umma_pipes = [
                pipeline.PipelineUmmaAsync.create(
                    barrier_storage=storage.umma_mbar.data_ptr() + g * 2,
                    num_stages=1,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    consumer_group=umma_cons_group,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
                for g in range(2)
            ]
            umma_pipeline_math = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar.data_ptr() + math_ring * 2,
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=umma_cons_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=288),
                allocator_warp_id=0,
                is_two_cta=False,
            )
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

            sKV = [
                smem.allocate_tensor(
                    element_type=self.a_dtype,
                    layout=a_smem_layout_staged.outer,
                    byte_alignment=128,
                    swizzle=a_smem_layout_staged.inner,
                )
                for _ in range(2)
            ]
            sQ = smem.allocate_tensor(
                element_type=self.b_dtype,
                layout=b_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=b_smem_layout_staged.inner,
            )
            sW = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((self.block_q, self.cols, 1), stride=(self.cols, 1, self.block_q * self.cols)),
                byte_alignment=128,
            )
            vec_layout = cute.make_layout(
                (BLOCK_KV, self.num_kv_stages, 2, 2),
                stride=(1, BLOCK_KV, BLOCK_KV * self.num_kv_stages, 2 * BLOCK_KV * self.num_kv_stages),
            )
            sVec = smem.allocate_tensor(element_type=Float32, layout=vec_layout, byte_alignment=128)
            sPair = smem.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((self.block_q // 2, self.pair_floats), stride=(self.pair_floats, 1)),
                byte_alignment=16,
            )
            sWin = smem.allocate_tensor(
                element_type=Int32, layout=cute.make_layout((self.block_q, 2), stride=(2, 1)), byte_alignment=16
            )

            thr_mma = tiled_mma.get_slice(0)
            gA = cute.local_tile(
                cute.domain_offset((kv_start, 0, 0), mA_mkl),
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            tCgA = thr_mma.partition_A(gA)
            tA = [
                cpasync.tma_partition(
                    tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sKV[g], 0, 3), cute.group_modes(tCgA, 0, 3)
                )
                for g in range(2)
            ]
            gVec = [cute.local_tile(cute.domain_offset((kv_start,), m), (BLOCK_KV,), (None,)) for m in (mS, mN)]
            tVec = [
                [
                    cpasync.tma_partition(
                        atom, 0, cute.make_layout(1), cute.group_modes(sVec[(None, None, which, g)], 0, 1), gVec[which]
                    )
                    for which, atom in enumerate((tma_atom_s, tma_atom_n))
                ]
                for g in range(2)
            ]
            gB = cute.local_tile(
                cute.domain_offset((tile * self.N, 0, 0), mB_nkl),
                cute.slice_(self.mma_tiler, (0, None, None)),
                (None, None, None),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3)
            )
            gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), (self.block_q, self.cols), (None, None))
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(gW, 0, 2)
            )

            tCrA = [tiled_mma.make_fragment_A(sKV[g]) for g in range(2)]
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
            epi_sub_mn = (epi_tile[0], self.N // NUM_EPI_SUBTILES)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype,
                (epi_sub_mn[0], 16), False,
            )

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            if (warp_idx == 8) | (warp_idx == 9):
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                group = warp_idx - 8
                kv_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_kv_stages)
                if group == 0:
                    q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                    q_pipeline.producer_acquire(q_prod)
                    q_bar = q_pipeline.producer_get_barrier(q_prod)
                    cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                    cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
                for g in cutlass.range_constexpr(2):
                    if group == g:
                        tAsA, tAgA = tA[g]
                        for pair in cutlass.range(num_pairs, unroll=1):
                            kv_pipes[g].producer_acquire(kv_prod)
                            bar = kv_pipes[g].producer_get_barrier(kv_prod)
                            stage = kv_prod.index
                            blk = pair * NUM_MATH_WG + g
                            cute.copy(tma_atom_a, tAgA[(None, blk, 0, 0)], tAsA[(None, stage)], tma_bar_ptr=bar)
                            for which, atom in enumerate((tma_atom_s, tma_atom_n)):
                                tVs, tVg = tVec[g][which]
                                cute.copy(atom, tVg[(None, blk)], tVs[(None, stage)], tma_bar_ptr=bar)
                            kv_prod.advance()
            elif warp_idx == 10:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
                named_barrier_sync_unaligned(288)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                acc_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[(None, None, None, 0)]
                acc_1 = cute.make_tensor(tmem_ptr + self.num_tmem_alloc_cols, tCtAcc_fake.layout)[(None, None, None, 0)]
                q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                kv_cons_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
                kv_cons_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
                umma_prod_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                umma_prod_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                num_k = cute.size(tCrA[0].shape[2])
                q_pipeline.consumer_wait(q_cons)
                for pair in cutlass.range(num_pairs, unroll=1):
                    ready = kv_pipes[0].consumer_try_wait(kv_cons_0)
                    empty = umma_pipes[0].producer_try_acquire(umma_prod_0)
                    kv_pipes[0].consumer_wait(kv_cons_0, ready)
                    umma_pipes[0].producer_acquire(umma_prod_0, empty)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(num_k):
                        cute.gemm(tiled_mma, acc_0, tCrA[0][None, None, k, kv_cons_0.index], tCrB[None, None, k, 0], acc_0)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons_0.advance()
                    umma_pipes[0].producer_commit(umma_prod_0)
                    umma_prod_0.advance()

                    ready = kv_pipes[1].consumer_try_wait(kv_cons_1)
                    empty = umma_pipes[1].producer_try_acquire(umma_prod_1)
                    kv_pipes[1].consumer_wait(kv_cons_1, ready)
                    umma_pipes[1].producer_acquire(umma_prod_1, empty)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range_constexpr(num_k):
                        cute.gemm(tiled_mma, acc_1, tCrA[1][None, None, k, kv_cons_1.index], tCrB[None, None, k, 0], acc_1)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons_1.advance()
                    umma_pipes[1].producer_commit(umma_prod_1)
                    umma_prod_1.advance()
            elif warp_idx < 8:
                self._run_math_warp(
                    num_pairs,
                    kv_start,
                    num_kv,
                    cover_lo,
                    cover_hi,
                    split_lo,
                    split_hi,
                    query_start,
                    tile,
                    num_q,
                    seq_len_kv,
                    mask_blocks,
                    copy_atom_t2r,
                    epi_sub_mn,
                    kv_pipeline_math,
                    q_pipeline,
                    sVec,
                    sW,
                    sPair,
                    sWin,
                    tCtAcc_fake,
                    tidx,
                    tmem,
                    umma_pipeline_math,
                    warpgroup_idx,
                    mRowStart,
                    mRowEnd,
                    mOrigin,
                    mInvDelta,
                    mThBucket,
                    mAllowance,
                    mMask,
                )
            else:
                cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self,
        num_pairs,
        kv_start,
        num_kv,
        cover_lo,
        cover_hi,
        split_lo,
        split_hi,
        query_start,
        tile,
        num_q,
        seq_len_kv,
        mask_blocks,
        copy_atom_t2r,
        epi_sub_mn,
        kv_pipeline_math,
        q_pipeline,
        sVec,
        sW,
        sPair,
        sWin,
        tCtAcc_fake,
        tidx,
        tmem,
        umma_pipeline_math,
        warpgroup_idx,
        mRowStart,
        mRowEnd,
        mOrigin,
        mInvDelta,
        mThBucket,
        mAllowance,
        mMask,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        lane = cute.arch.lane_idx()
        quad = cute.arch.make_warp_uniform(cute.arch.warp_idx() % 4)
        tmem.allocate(2 * self.num_tmem_alloc_cols)
        named_barrier_sync_unaligned(288)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr + warpgroup_idx * self.num_tmem_alloc_cols, tCtAcc_fake.layout)
        epi_tidx = tidx % 128
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, None)], epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)[(None, None, None, None, None, 0)]
        cC = cute.make_identity_tensor(epi_sub_mn)
        tTR_rAcc = cute.make_rmem_tensor_like(thr_copy_t2r.partition_D(cC), self.acc_dtype)
        m_coord = thr_copy_t2r.partition_D(cC)[0][0]

        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        q_pipeline.consumer_wait(q_cons)
        if tidx < self.block_q * self.cols:
            r = tidx // self.cols
            c = tidx % self.cols
            w = sW[(r, c, 0)]
            if c < self.cols - 2:
                w = w * Float32(0.5)
            sPair[(r // 2, 2 * c + r % 2)] = w
        if tidx < self.block_q:
            row_q = query_start + tidx
            row = min(row_q, num_q - 1)
            sPair[(tidx // 2, 2 * self.cols + tidx % 2)] = mAllowance[row]
            sPair[(tidx // 2, 2 * self.cols + 2 + tidx % 2)] = loose_raw_gate(
                mInvDelta[row], mOrigin[row], mThBucket[row]
            )
            s = seq_len_kv
            e = Int32(0)
            if row_q < num_q:
                s = max(min(mRowStart[row_q], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row_q], seq_len_kv), split_hi)
            sWin[(tidx, 0)] = s
            sWin[(tidx, 1)] = e
        cute.arch.sync_warp()
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()

        # Cache the first 14 retained heads' pair weights (4 pairs x 28 = 112 floats) in registers.
        w_reg = cute.make_rmem_tensor(cute.make_layout((4, 28), stride=(28, 1)), Float32)
        for pp in cutlass.range_constexpr(4):
            for wi in cutlass.range_constexpr(28):
                w_reg[(pp, wi)] = sPair[(pp, wi)]

        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
        umma_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        tile_base = (tile * mask_blocks) * 4 + quad
        for pair in cutlass.range(num_pairs, unroll=1):
            j = pair * NUM_MATH_WG + warpgroup_idx
            umma_pipeline_math.consumer_wait(umma_cons)
            cute.arch.sync_warp()
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc)
            scale = sVec[(m_coord, kv_cons.index, 0, warpgroup_idx)]
            norm = sVec[(m_coord, kv_cons.index, 1, warpgroup_idx)]
            # UMMA already finished reading sKV; release KV stage right after scale/norm are in registers.
            cute.arch.sync_warp()
            kv_pipeline_math.consumer_release(kv_cons)
            kv_cons.advance()
            cute.arch.fence_view_async_tmem_load()
            blk_lo = kv_start + j * BLOCK_KV
            kv_idx = blk_lo + m_coord
            interior = (blk_lo >= cover_lo) & (blk_lo + BLOCK_KV <= cover_hi)
            surv = cutlass.Boolean(False)
            for e in cutlass.range_constexpr(NUM_EPI_SUBTILES):
                if e > 0:
                    cute.arch.fence_view_async_tmem_load()
                if e == NUM_EPI_SUBTILES - 1:
                    umma_pipeline_math.consumer_release(umma_cons)
                    umma_cons.advance()
                acc = tTR_rAcc.load()
                if e + 1 < NUM_EPI_SUBTILES:
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, e + 1)], tTR_rAcc)
                keep = self._pair_bound(acc, e, sPair, w_reg, scale, norm)
                for half_row in cutlass.range_constexpr(2):
                    r = 2 * e + half_row
                    in_win = interior | ((kv_idx >= sWin[(r, 0)]) & (kv_idx < sWin[(r, 1)]))
                    surv = surv | (keep[half_row] & in_win)
            word = cute.arch.vote_ballot_sync(surv)
            if (lane == 0) & (j < num_kv):
                gblock = kv_start // BLOCK_KV + j
                base = mMask.iterator.toint() + cutlass.Int64(tile_base + gblock * 4) * 4
                store_global_b32(base, word)
            cute.arch.sync_warp()

        q_pipeline.consumer_release(q_cons)
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _pair_bound(self, acc, p, sPair, w_reg, scale, norm):
        """Whether each row of pair p keeps the key: bound > gate (NaN keeps)."""
        cols = self.cols

        def x(c):
            return (acc[2 * c], acc[2 * c + 1])

        def pos2(v):  # 2 relu(v), exactly
            return cute.arch.add_packed_f32x2(v, (abs_f32(v[0]), abs_f32(v[1])), rnd=_RND_RN)

        def w(i: cutlass.Constexpr):
            if cutlass.const_expr(i < 28):
                return w_reg[(p, i)]
            return sPair[(p, i)]

        acc0 = (Float32(0.0), Float32(0.0))
        acc1 = (Float32(0.0), Float32(0.0))
        for c in cutlass.range_constexpr(0, cols - 2, 2):
            acc0 = cute.arch.fma_packed_f32x2(pos2(x(c)), (w(2 * c), w(2 * c + 1)), acc0, rnd=_RND_RN)
            acc1 = cute.arch.fma_packed_f32x2(pos2(x(c + 1)), (w(2 * c + 2), w(2 * c + 3)), acc1, rnd=_RND_RN)
        k = 2 * (cols - 2)
        agg = cute.arch.fma_packed_f32x2(
            x(cols - 1),
            (w(k + 2), w(k + 3)),
            cute.arch.mul_packed_f32x2(x(cols - 2), (w(k), w(k + 1)), rnd=_RND_RN),
            rnd=_RND_RN,
        )
        reduced = cute.arch.add_packed_f32x2(
            cute.arch.add_packed_f32x2(acc0, acc1, rnd=_RND_RN),
            (-fmax(agg[0], Float32(0.0)), -fmax(agg[1], Float32(0.0))),
            rnd=_RND_RN,
        )
        a = 2 * cols
        bound = cute.arch.fma_packed_f32x2(
            (w(a), w(a + 1)),
            (norm, norm),
            cute.arch.mul_packed_f32x2((scale, scale), reduced, rnd=_RND_RN),
            rnd=_RND_RN,
        )
        gates = (w(a + 2), w(a + 3))
        return [(bound[i] > gates[i]) | (bound[i] != bound[i]) | (gates[i] != gates[i]) for i in (0, 1)]
