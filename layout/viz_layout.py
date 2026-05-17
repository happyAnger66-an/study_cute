from cutlass import cute

try:
    from cute_viz import display_tv_layout

    @cute.jit
    def visualize():
        # Create and render a layout to file
        # layout = cute.make_layout( ((16,16),(256,2)), stride=((512,8192),(1,256)))
        # display_layout(layout)

        tv_layout = cute.make_layout(((32, 4), (8, 4)), stride=((128, 4), (16, 1)))
        display_tv_layout(tv_layout, (16, 256))

        thr_block_layout = cute.make_layout((16, 256), stride=(512, 1))
        print(cute.composition(thr_block_layout, tv_layout))

    visualize()
except ImportError:
    print("cute_viz is not installed")
    pass