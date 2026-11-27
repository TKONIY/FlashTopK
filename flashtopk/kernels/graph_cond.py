# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""CUDA Graph device-side conditional nodes (CUDA 12.4+) via pure driver API + embedded PTX.

When ``Selector.run()`` is captured inside a CUDA Graph, untaken adaptive-pruning branches
(scan vs 16-col vs 24-col) and unneeded recovery branches (healthy vs per-row vs whole-call)
are wrapped in native ``CU_GRAPH_COND_TYPE_SWITCH`` nodes so untaken branches execute ZERO
kernel launches during graph replay (0.000 ms overhead), while still capturing and replaying
all branches bit-exactly when device flags change across replays.

In eager mode (not capturing) or on drivers without conditional-node support, helpers return
``False`` cleanly so callers run their standard device-gated eager path.
"""

from __future__ import annotations

import ctypes
from typing import Callable, Optional, Sequence

import cuda.bindings.driver as cu
import torch

_PTX = b"""
.version 8.4
.target sm_90
.address_size 64

.extern .func cudaGraphSetConditional
(
    .param .b64 cudaGraphSetConditional_param_0,
    .param .b32 cudaGraphSetConditional_param_1
);

.visible .entry set_cond_switch_i32(
    .param .u64 p_handle,
    .param .u64 p_flag
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    ld.param.b64 %rd1, [p_handle];
    ld.param.b64 %rd2, [p_flag];
    cvta.to.global.u64 %rd3, %rd2;
    ld.global.b32 %r1, [%rd3];
    {
        .param .b64 param0;
        .param .b32 param1;
        st.param.b64 [param0], %rd1;
        st.param.b32 [param1], %r1;
        call.uni cudaGraphSetConditional, (param0, param1);
    }
    ret;
}

.visible .entry set_cond_recovery_switch(
    .param .u64 p_handle,
    .param .u64 p_health
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    ld.param.b64 %rd1, [p_handle];
    ld.param.b64 %rd2, [p_health];
    cvta.to.global.u64 %rd3, %rd2;
    ld.global.b64 %rd4, [%rd3];       // health[0] H_OVERFLOW
    ld.global.b64 %rd5, [%rd3+8];     // health[1] H_UNDERFILL
    or.b64 %rd4, %rd4, %rd5;
    setp.ne.s64 %p1, %rd4, 0;         // failed
    ld.global.b64 %rd5, [%rd3+56];    // health[7] H_BAD
    setp.gt.s64 %p2, %rd5, 0;         // bad > 0
    setp.le.s64 %p3, %rd5, 64;        // bad <= RECOVERY_ROWS (64)
    and.pred %p4, %p2, %p3;           // per_row (0 < bad <= 64)
    // 0: healthy (no-op), 1: per-row recovery, 2: whole-call recovery
    selp.b32 %r2, 1, 2, %p4;
    selp.b32 %r1, %r2, 0, %p1;
    {
        .param .b64 param0;
        .param .b32 param1;
        st.param.b64 [param0], %rd1;
        st.param.b32 [param1], %r1;
        call.uni cudaGraphSetConditional, (param0, param1);
    }
    ret;
}
"""

_STATE: dict = {}


def _check(res):
    err = res[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed with status {err}")
    return res[1:] if len(res) > 2 else (res[1] if len(res) == 2 else None)


def _device_state(device_index: int):
    state = _STATE.get(device_index)
    if state is not None:
        return state
    if not hasattr(cu, "cuGraphConditionalHandleCreate") or not hasattr(cu, "cuStreamBeginCaptureToGraph"):
        _STATE[device_index] = False
        return False
    try:
        with torch.cuda.device(device_index):
            ctx = _check(cu.cuCtxGetCurrent())
            mod = _check(cu.cuModuleLoadData(_PTX))
            fn_switch = _check(cu.cuModuleGetFunction(mod, b"set_cond_switch_i32"))
            fn_rec = _check(cu.cuModuleGetFunction(mod, b"set_cond_recovery_switch"))
            side = torch.cuda.Stream(device=device_index)
        state = (ctx, mod, fn_switch, fn_rec, side)
        _STATE[device_index] = state
        return state
    except Exception:
        _STATE[device_index] = False
        return False


def is_capturing(device_index: int) -> bool:
    """True when the current CUDA stream on `device_index` is actively capturing a CUDA Graph
    and conditional graph nodes are supported."""
    if not torch.cuda.is_current_stream_capturing():
        return False
    return bool(_device_state(device_index))


def _emit_switch(
    device_index: int,
    setter_fn,
    data_ptr: int,
    branches: Sequence[Optional[Callable[[], None]]],
) -> bool:
    if not torch.cuda.is_current_stream_capturing():
        return False
    state = _device_state(device_index)
    if not state:
        return False
    ctx, _mod, _fn_sw, _fn_rec, side = state
    stream_ptr = torch.cuda.current_stream(device_index).cuda_stream
    info = _check(cu.cuStreamGetCaptureInfo(stream_ptr))
    if info[0] != cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        return False
    graph = info[2]
    handle = _check(cu.cuGraphConditionalHandleCreate(graph, ctx, 0, cu.CU_GRAPH_COND_ASSIGN_DEFAULT))

    # Launch the 1-thread setter kernel on the main capturing stream so it depends on prior work.
    a0 = ctypes.c_uint64(int(handle))
    a1 = ctypes.c_uint64(int(data_ptr))
    args = (ctypes.c_void_p * 2)(ctypes.addressof(a0), ctypes.addressof(a1))
    _check(cu.cuLaunchKernel(setter_fn, 1, 1, 1, 1, 1, 1, 0, stream_ptr, args, 0))

    info2 = _check(cu.cuStreamGetCaptureInfo(stream_ptr))
    graph, deps = info2[2], info2[3]

    params = cu.CUgraphNodeParams()
    params.type = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    params.conditional.handle = handle
    params.conditional.type = cu.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_SWITCH
    params.conditional.size = len(branches)
    params.conditional.ctx = ctx
    cond_node = _check(cu.cuGraphAddNode(graph, deps, None, len(deps), params))
    child_graphs = params.conditional.phGraph_out

    for idx, fn in enumerate(branches):
        if fn is None:
            continue
        _check(
            cu.cuStreamBeginCaptureToGraph(
                side.cuda_stream,
                child_graphs[idx],
                None,
                None,
                0,
                cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        )
        try:
            with torch.cuda.stream(side):
                fn()
        finally:
            _check(cu.cuStreamEndCapture(side.cuda_stream))

    _check(
        cu.cuStreamUpdateCaptureDependencies(
            stream_ptr,
            [cond_node],
            None,
            1,
            cu.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
        )
    )
    return True


def switch_i32(flag: torch.Tensor, branches: Sequence[Optional[Callable[[], None]]]) -> bool:
    """If capturing a CUDA Graph, emit a SWITCH conditional node selecting ``branches[flag[0]]``
    and return True; otherwise return False."""
    if not torch.cuda.is_current_stream_capturing():
        return False
    state = _device_state(flag.device.index)
    if not state:
        return False
    return _emit_switch(flag.device.index, state[2], flag.data_ptr(), branches)


def recovery_switch(
    health: torch.Tensor,
    per_row_fn: Callable[[], None],
    whole_call_fn: Callable[[], None],
) -> bool:
    """If capturing a CUDA Graph, emit a 3-way SWITCH conditional node:
      0: healthy (empty child graph, 0 launches),
      1: per-row recovery (0 < bad <= 64),
      2: whole-call recovery (failed and (bad <= 0 or bad > 64)).
    Returns True if emitted, else False."""
    if not torch.cuda.is_current_stream_capturing():
        return False
    state = _device_state(health.device.index)
    if not state:
        return False
    return _emit_switch(health.device.index, state[3], health.data_ptr(), (None, per_row_fn, whole_call_fn))
