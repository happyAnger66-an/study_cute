"""Host-side (CPU) components of the FMHA kernel.

- :mod:`fmha.host.config`         configuration class + host validation
- :mod:`fmha.host.launcher`       ``__call__`` / ``__call_vit__`` JIT entry points
- :mod:`fmha.host.tensor_layout`  tensor padding / dtype / layout helpers
- :mod:`fmha.host.numpy_ref`      numpy reference attention implementations
"""
