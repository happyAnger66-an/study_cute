from cutlass import cute

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # --------------------------------
    # slice for thread-block level view
    # --------------------------------
    blk_coord = ((None, None), bidx)

    # logical coord -> address
    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address
    blkB = gB[blk_coord]  # (TileM, TileN) -> physical address
    blkC = gC[blk_coord]  # (TileM, TileN) -> physical address

    # --------------------------------
    # compose for thread-index & value-index to physical mapping
    # --------------------------------
    # blockA:    (TileM, TileN) -> physical address
    # tv_layout: (tid, vid)     -> (TileM, TileN)
    # tidfrgA = blkA o tv_layout
    # tidfrgA:   (tid, vid) -> physical address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print("Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    # --------------------------------
    # slice for thread-level view
    # --------------------------------
    # `None` represent slice of the entire per-thread data
    thr_coord = (tidx, None)
    # thr_coord = (tidx, cute.repeat_like(None, gA.shape[1]))

    # slice for threads: vid -> address
    thrA = tidfrgA[thr_coord]  # (V) -> physical address
    thrB = tidfrgB[thr_coord]  # (V) -> physical address
    thrC = tidfrgC[thr_coord]  # (V) -> physical address

    thrC[None] = thrA.load() + thrB.load()

import torch
from cutlass.cute.runtime import from_dlpack

M, N = 2048, 2048

@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA layout: (M, N):(N, 1)
    # TV layout map thread & value index to (64, 512) logical tile
    #  - contiguous thread index maps to mode-1 because input layout is contiguous on
    #     mode-1 for coalesced load-store
    #  - each thread load contiguous 16 bytes each row and load 16 rows
    coalesced_ldst_bytes = 16

    # Compile time validation: expect same element type for all input tensors
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    print(f"[DSL INFO] Thr Layout: {thr_layout}")
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    print(f"[DSL INFO] Val Layout: {val_layout}")
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    print(f"[DSL INFO] Recasted Val Layout: {val_layout}")
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print("Tiled Input Tensors:")
    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    # Launch the kernel asynchronously
    # Async token(s) can also be specified as dependencies
    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
elementwise_add_(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c, a + b)