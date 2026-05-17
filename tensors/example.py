import cutlass
from cutlass import cute

@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(1, 8))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)

if __name__ == "__main__":
    import torch

    from cutlass.torch import dtype as torch_dtype
    import cutlass.cute.runtime as cute_rt

    a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
    ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())

    create_tensor_from_ptr(ptr_a)