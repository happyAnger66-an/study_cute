"""Device-side (GPU) components of the FMHA kernel.

- :mod:`fmha.device.kernel`             top-level ``@cute.kernel`` warp dispatcher
- :mod:`fmha.device.warp_load`          producer warp: TMA Q/K/V loads
- :mod:`fmha.device.warp_mma`           tensor-core warp: QK / PV ``cute.gemm``
- :mod:`fmha.device.warp_softmax`       softmax warps (stage 0 / 1) + per-step helper
- :mod:`fmha.device.warp_correction`    correction + epilog rescaling
- :mod:`fmha.device.warp_epilogue`      epilogue warp: TMA O store
"""
