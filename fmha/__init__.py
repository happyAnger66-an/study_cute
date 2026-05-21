"""Blackwell SM100 Fused Multi-Head Attention (CuTe DSL).

Package layout:
- host/     CPU-side configuration, launchers, tensor helpers, numpy reference
- device/   GPU-side @cute.kernel and per-warp @cute.jit functions

The top-level class :class:`BlackwellFusedMultiHeadAttentionForward` is defined
in :mod:`fmha.host.config`. Device-side ``@cute.kernel`` / ``@cute.jit`` methods
are defined in :mod:`fmha.device` modules and bound onto the class here so the
class always sees them as ``self.kernel`` / ``self.softmax`` / etc.
"""

from fmha.host.config import BlackwellFusedMultiHeadAttentionForward
from fmha.host.launcher import _call_llm, _call_vit
from fmha.device.kernel import kernel as _device_kernel
from fmha.device.warp_load import load_warp_body as _load_warp_body
from fmha.device.warp_mma import mma_warp_body as _mma_warp_body
from fmha.device.warp_softmax import (
    softmax as _softmax_method,
    softmax_step as _softmax_step_method,
)
from fmha.device.warp_correction import (
    correction_warp_body as _correction_warp_body,
    correction_rescale as _correction_rescale_method,
    correction_epilog as _correction_epilog_method,
)
from fmha.device.warp_epilogue import epilogue_warp_body as _epilogue_warp_body

# Bind @cute.jit functions as methods so the DSL treats the cfg/self as a
# static Python receiver instead of a dynamic value (which would fail to
# flatten because the cfg object isn't a Numeric).
BlackwellFusedMultiHeadAttentionForward.__call__ = _call_llm
BlackwellFusedMultiHeadAttentionForward.__call_vit__ = _call_vit
BlackwellFusedMultiHeadAttentionForward.kernel = _device_kernel
BlackwellFusedMultiHeadAttentionForward.load_warp_body = _load_warp_body
BlackwellFusedMultiHeadAttentionForward.mma_warp_body = _mma_warp_body
BlackwellFusedMultiHeadAttentionForward.softmax = _softmax_method
BlackwellFusedMultiHeadAttentionForward.softmax_step = _softmax_step_method
BlackwellFusedMultiHeadAttentionForward.correction_warp_body = _correction_warp_body
BlackwellFusedMultiHeadAttentionForward.correction_rescale = _correction_rescale_method
BlackwellFusedMultiHeadAttentionForward.correction_epilog = _correction_epilog_method
BlackwellFusedMultiHeadAttentionForward.epilogue_warp_body = _epilogue_warp_body

__all__ = ["BlackwellFusedMultiHeadAttentionForward"]
