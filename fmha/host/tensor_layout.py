"""Host-side tensor layout / dtype helpers.

All helpers here live on the CPU side. They prepare CuPy / NumPy buffers,
mark CuTe tensor layout properties, and convert between cutlass / cupy dtypes
so the @cute.jit launchers only deal with already-typed cute.Tensor objects.
"""

from typing import Tuple

import cupy as cp
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------
def cutlass_to_cupy_dtype(cutlass_dtype):
    """Map a cutlass scalar dtype to its CuPy storage dtype.

    FP8 / Int4 are stored as packed bytes (uint8); FP32 / FP16 map directly.
    """
    if cutlass_dtype == cutlass.Float16:
        return cp.float16
    if cutlass_dtype in (cutlass.Float32, cutlass.cute.typing.Float32):
        return cp.float32
    if (cutlass_dtype.is_float and cutlass_dtype.width <= 8) or (
        cutlass_dtype.is_integer and cutlass_dtype.width == 4
    ):
        return cp.uint8
    raise ValueError(f"Unsupported dtype for CuPy: {cutlass_dtype}")


def get_leading_dim(cp_array):
    """Return the index of the most-contiguous dim (stride == itemsize)."""
    for i, s in enumerate(cp_array.strides):
        if s == cp_array.itemsize:
            return i
    return len(cp_array.shape) - 1


# ---------------------------------------------------------------------------
# Tensor factory
# ---------------------------------------------------------------------------
def create_and_pad_tensor(
    shape: Tuple[int, ...],
    padding: Tuple[int, ...],
    dtype,
    is_dynamic_layout: bool = True,
    export_only: bool = False,
):
    """Allocate a (padded) GPU tensor and wrap it for both CuTe and numpy.

    Returns ``(f32_ref_np, cute_tensor, dtype_gpu, dtype_gpu_full, f32_gpu_full)``.
    The trailing buffers prevent CuPy from GC'ing the underlying storage while
    the slice view is still in use.
    """
    shape_ = tuple(map(lambda x, y: x + y, shape, padding))

    if export_only:
        f32_gpu_full = cp.zeros(shape_, dtype=cp.float32)
    else:
        min_val = -2 if dtype.is_float or dtype.signed else 0
        f32_gpu_full = cp.random.randint(min_val, 2, shape_).astype(cp.float32)

    cp_dtype = cutlass_to_cupy_dtype(dtype)
    is_narrow = (dtype.is_float and dtype.width <= 8) or (
        dtype.is_integer and dtype.width == 4
    )
    dtype_gpu_full = cp.empty(shape_, dtype=cp_dtype)

    if is_narrow:
        f32_cute = from_dlpack(f32_gpu_full)
        if is_dynamic_layout:
            f32_cute = f32_cute.mark_layout_dynamic(
                leading_dim=get_leading_dim(f32_gpu_full)
            )
        dtype_cute_full = from_dlpack(dtype_gpu_full, assumed_align=16)
        dtype_cute_full.element_type = dtype
        if is_dynamic_layout:
            dtype_cute_full = dtype_cute_full.mark_layout_dynamic(
                leading_dim=get_leading_dim(dtype_gpu_full)
            )
        cute.testing.convert(f32_cute, dtype_cute_full)
    else:
        dtype_gpu_full[:] = f32_gpu_full.astype(cp_dtype)

    slices = tuple(slice(s, e) for s, e in zip(padding, shape_))
    dtype_gpu = dtype_gpu_full[slices]
    f32_gpu = f32_gpu_full[slices]

    cute_tensor = from_dlpack(dtype_gpu, assumed_align=16)
    cute_tensor.element_type = dtype

    f32_ref = f32_gpu.get()
    return (f32_ref, cute_tensor, dtype_gpu, dtype_gpu_full, f32_gpu_full)


# ---------------------------------------------------------------------------
# Layout dynamism markers
# ---------------------------------------------------------------------------
def mark_bshd_dynamic(tensor):
    """Mark a contiguous [B, S, H, D] tensor's first three dims as dynamic."""
    so = (0, 1, 2, 3)
    return (
        tensor.mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )


def mark_shd_dynamic(tensor):
    """Mark a packed [total_S, H, D] tensor's first two dims as dynamic."""
    so = (0, 1, 2)
    return (
        tensor.mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
    )


def mark_kv_cache_dynamic(tensor):
    """Mark a contiguous [B, 2, H_kv, S, D] KV cache tensor as dynamic.

    Modes 0/2/3 (B / H_kv / S) are dynamic; mode 1 (the K/V selector, 2) is
    kept static so the kernel can fold its stride at compile time.
    """
    so = (0, 1, 2, 3, 4)
    return (
        tensor.mark_layout_dynamic(leading_dim=4)
        .mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
        .mark_compact_shape_dynamic(mode=3, stride_order=so)
    )


def mark_1d_dynamic(tensor):
    """Mark a 1-D tensor (e.g. ``cu_seqlens``) as dynamic on its single axis."""
    return tensor.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )
