"""Blackwell SM100 FMHA — thin shim around the modular :mod:`fmha` package.

The legacy single-file implementation was split into:

- :mod:`fmha.host.config`         configuration class
- :mod:`fmha.host.launcher`       ``__call__`` / ``__call_vit__`` JIT launchers
- :mod:`fmha.host.tensor_layout`  tensor padding / dtype / layout helpers
- :mod:`fmha.host.numpy_ref`      numpy reference attention
- :mod:`fmha.device.kernel`       top-level ``@cute.kernel`` warp dispatcher
- :mod:`fmha.device.warp_load`    producer warp body
- :mod:`fmha.device.warp_mma`     tensor-core warp body (D=256 hang locus)
- :mod:`fmha.device.warp_softmax` softmax warps + per-step inner loop
- :mod:`fmha.device.warp_correction`  correction warpgroup + rescale/epilog
- :mod:`fmha.device.warp_epilogue`    epilogue warp (TMA O store)
- :mod:`fmha.runner`              CLI + ``run`` / multi-round prefill test

This file preserves the legacy ``python3 fmha/fmha.py --q_shape ...`` entry
point so existing scripts keep working.
"""

import os
import sys

if __name__ == "__main__":
    # Allow ``python3 fmha/fmha.py`` to find ``fmha_helpers``/``fmha`` (parent).
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, ".."))

from fmha import BlackwellFusedMultiHeadAttentionForward  # noqa: E402,F401
from fmha.runner import (  # noqa: E402,F401
    main,
    run,
    run_llm_multi_round_prefill_test,
)


if __name__ == "__main__":
    main()
