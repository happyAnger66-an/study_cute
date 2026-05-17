from cutlass import cute

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