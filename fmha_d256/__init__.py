"""CUTLASS official d=256 mixed-input FMHA prefill (study_cute package).

Package layout mirrors ``fmha/``:

- ``host/``     CPU-side config, launcher, torch reference
- ``device/``   GPU-side ``@cute.kernel`` and per-warp ``@cute.jit`` bodies
- ``runner.py`` test driver / benchmark entry
"""

from fmha_d256.host.config import MixedInputFusedMultiHeadAttentionPrefillD256
from fmha_d256.host.launcher import launch as _launch
from fmha_d256.host.launcher import launch_homo as _launch_homo
from fmha_d256.host.launcher import _call_llm as _call_llm
from fmha_d256.device.kernel import kernel as _device_kernel
from fmha_d256.device.kernel import kernel_homo as _device_kernel_homo
from fmha_d256.device.warp_load import load_warp_body as _load_warp_body
from fmha_d256.device.warp_mma import (
    mma_warp_body as _mma_warp_body,
    mma_pv as _mma_pv,
)
from fmha_d256.device.warp_softmax import (
    softmax_warp_body as _softmax_warp_body,
    softmax_step as _softmax_step,
    store_sum as _store_sum,
)
from fmha_d256.device.warp_correction import (
    correction_warp_body as _correction_warp_body,
    correction_rescale as _correction_rescale,
    correction_epilog as _correction_epilog,
)
from fmha_d256.device.warp_transform import transform_warp_body as _transform_warp_body
from fmha_d256.runner import run

# Bind @cute.jit / @cute.kernel callables as methods so the DSL treats ``self``
# as a static Python receiver (same pattern as fmha/__init__.py).
MixedInputFusedMultiHeadAttentionPrefillD256.__call__ = _launch
MixedInputFusedMultiHeadAttentionPrefillD256.launch_homo = _launch_homo
MixedInputFusedMultiHeadAttentionPrefillD256.call_llm = _call_llm
MixedInputFusedMultiHeadAttentionPrefillD256.kernel = _device_kernel
MixedInputFusedMultiHeadAttentionPrefillD256.kernel_homo = _device_kernel_homo
MixedInputFusedMultiHeadAttentionPrefillD256.load_warp_body = _load_warp_body
MixedInputFusedMultiHeadAttentionPrefillD256.mma_warp_body = _mma_warp_body
MixedInputFusedMultiHeadAttentionPrefillD256.mma_pv = _mma_pv
MixedInputFusedMultiHeadAttentionPrefillD256.softmax_warp_body = _softmax_warp_body
MixedInputFusedMultiHeadAttentionPrefillD256.softmax_step = _softmax_step
MixedInputFusedMultiHeadAttentionPrefillD256.store_sum = _store_sum
MixedInputFusedMultiHeadAttentionPrefillD256.correction_warp_body = _correction_warp_body
MixedInputFusedMultiHeadAttentionPrefillD256.correction_rescale = _correction_rescale
MixedInputFusedMultiHeadAttentionPrefillD256.correction_epilog = _correction_epilog
MixedInputFusedMultiHeadAttentionPrefillD256.transform_warp_body = _transform_warp_body

__all__ = [
    "MixedInputFusedMultiHeadAttentionPrefillD256",
    "run",
    "run_llm_multi_round_prefill_test_d256",
]
