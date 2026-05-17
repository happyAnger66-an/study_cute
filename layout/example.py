import cutlass
from cutlass import cute

@cute.jit
def coalesce_example():
    """
    Demonstrates coalesce operation flattening and combining modes
    """
    layout = cute.make_layout(
        (2, (1, 6)), stride=(1, (cutlass.Int32(6), 2))
    )  # Dynamic stride
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    cute.printf(">?? Original: {}", layout)
    print(">>> Coalesced:", result)
    cute.printf(">?? Coalesced: {}", result)

@cute.jit
def composition_example():
    """
    Demonstrates basic layout composition R = A ◦ B
    """
    A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2))  # Dynamic stride
    B = cute.make_layout((4, 3), stride=(3, 1))
    R = cute.composition(A, B)

    # Print static and dynamic information
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Layout B:", B)
    cute.printf(">?? Layout B: {}", B)
    print(">>> Composition R = A ◦ B:", R)
    cute.printf(">?? Composition R: {}", R)

coalesce_example()
composition_example()