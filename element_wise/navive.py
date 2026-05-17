from cutlass import cute

# Basic Kernel Implementation
# ---------------------
# This is our first implementation of the elementwise add kernel.
# It follows a simple 1:1 mapping between threads and tensor elements.


@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,  # Input tensor A
    gB: cute.Tensor,  # Input tensor B
    gC: cute.Tensor,  # Output tensor C = A + B
):
    # Step 1: Get thread indices
    # ------------------------
    # CUDA threads are organized in a 3D grid of thread blocks
    # Here we only use the x-dimension for simplicity
    tidx, _, _ = cute.arch.thread_idx()  # Thread index within block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()  # Block index in grid (0 to grid_dim-1)
    bdim, _, _ = cute.arch.block_dim()  # Number of threads per block

    # Calculate global thread index
    # This gives each thread a unique ID across all blocks
    thread_idx = bidx * bdim + tidx  # Global thread ID

    # Step 2: Map thread index to tensor coordinates
    # -------------------------------------------
    # Each thread will process one element of the input tensors
    m, n = gA.shape  # Get tensor dimensions (M rows × N columns)

    # Convert linear thread index to 2D coordinates:
    # - ni: column index (0 to n-1)
    # - mi: row index (0 to m-1)
    ni = thread_idx % n  # Column index (faster varying dimension)
    mi = thread_idx // n  # Row index (slower varying dimension)

    # Step 3: Load and process data
    # ---------------------------
    # Load values from input tensors
    # The tensor layout automatically handles the conversion from
    # logical indices (mi, ni) to physical memory addresses
    a_val = gA[mi, ni]  # Load element from tensor A
    b_val = gB[mi, ni]  # Load element from tensor B

    # Step 4: Store result
    # ------------------
    # Write the sum back to the output tensor
    gC[mi, ni] = a_val + b_val

@cute.jit  # Just-in-time compilation decorator
def naive_elementwise_add(
    mA: cute.Tensor,  # Input tensor A
    mB: cute.Tensor,  # Input tensor B
    mC: cute.Tensor,  # Output tensor C
):
    # Configure kernel launch parameters
    # --------------------------------
    # Choose number of threads per block
    # 256 is a common choice as it:
    # - Allows good occupancy on most GPUs
    # - Is a multiple of 32 (warp size)
    # - Provides enough threads for latency hiding
    num_threads_per_block = 256

    # Get input dimensions
    m, n = mA.shape  # Matrix dimensions (M rows × N columns)

    # Create kernel instance
    kernel = naive_elementwise_add_kernel(mA, mB, mC)

    # Launch kernel with calculated grid dimensions
    # -------------------------------------------
    # Grid size calculation:
    # - Total elements: m * n
    # - Blocks needed: ceil(total_elements / threads_per_block)
    # - Using integer division here assumes m * n is multiple of threads_per_block
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),  # Number of blocks in x,y,z
        block=(num_threads_per_block, 1, 1),  # Threads per block in x,y,z
    )

def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    # Calculate metrics
    # ----------------
    dtype = a_.element_type

    # Calculate total bytes transferred:
    # - 2 reads (A and B) + 1 write (C)
    # - Each element is dtype.width bits
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    # Calculate achieved bandwidth
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")

import torch
from cutlass.cute.runtime import from_dlpack

# Test Setup
# ----------
# Define test dimensions
M, N = 16384, 8192  # Using large matrices to measure performance

# Create test data on GPU
# ----------------------
# Using float16 (half precision) for:
# - Reduced memory bandwidth requirements
# - Better performance on modern GPUs
a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input A
b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input B
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # Output buffer

# Calculate total elements for bandwidth calculations
num_elements = sum([a.numel(), b.numel(), c.numel()])

# Convert PyTorch tensors to CuTe tensors
# -------------------------------------
# from_dlpack creates CuTe tensor views of PyTorch tensors
# assumed_align=16 ensures proper memory alignment for vectorized access
a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C


# Compile the kernel for the specific input types
naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)

# Run the kernel
naive_elementwise_add_(a_, b_, c_)

# Verify Results
# -------------
# Compare our kernel output with PyTorch's native implementation
torch.testing.assert_close(c, a + b)  # Raises error if results don't match

print(f'verify results successfully!')

benchmark(naive_elementwise_add_, a_, b_, c_)