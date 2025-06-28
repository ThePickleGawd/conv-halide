"""
Generates depthwise separable convolutions in halide. Needs to be compiled to be used. See example below.

cl /LD /EHsc /std:c++17 sep_conv.py.cpp sep_conv.o ^
    /I C:\Users\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\halide\include ^
    /I C:\User\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\pybind11\include ^
    /I C:\Users\dylan\AppData\Roaming\uv\python\cpython-3.9.21-windows-x86_64-none\Include ^
    /link ^
    /LIBPATH:C:\Users\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\halide\lib ^
    /LIBPATH:C:\Users\dylan\AppData\Roaming\uv\python\cpython-3.9.21-windows-x86_64-none\libs ^
    python39.lib ^
    /OUT:sep_conv.pyd

Note: np and halide have dimensions flipped
"""

import halide as hl
import numpy as np


def main():
    # Vars and Funcs
    dw = hl.Func("dw")
    pw = hl.Func("pw")
    x, y, c, n = hl.Var("x"), hl.Var("y"), hl.Var("c"), hl.Var("n")

    # Params
    input = hl.ImageParam(hl.Float(32), 4, "input_feature_map")  # (W, H, C_in, N)
    kernel = hl.ImageParam(hl.Float(32), 4, "kernel")  # (KW, KH, C_in, N)
    pw_filter = hl.ImageParam(hl.Float(32), 3, "pw_filter")  # (C_out, C_in, N)
    bias = hl.ImageParam(hl.Float(32), 2, "bias")

    bounded = hl.BoundaryConditions.repeat_edge(input)

    # dw_conv
    r = hl.RDom([(0, kernel.dim(0).extent()), (0, kernel.dim(1).extent())])  # KW, KH
    dw[x, y, c, n] = hl.Expr(0.0)
    dw[x, y, c, n] += bounded[x - r.x + 1, y - r.y + 1, c, n] * kernel[r.x, r.y, c, n]

    # pw_conv
    rc = hl.RDom([(0, input.dim(2).extent())])
    pw[x, y, c, n] = bias[c, n]
    pw[x, y, c, n] += dw[x, y, rc, n] * pw_filter[c, rc, n]

    # sep_conv output (we can add ReLU here if we want)
    sep_conv = pw

    test = False
    if test:
        N, H, W = 2, 8, 8
        C_in, C_out = 2, 3
        KH, KW = 3, 3

        input_np = np.ones((N, C_in, H, W), dtype=np.float32)  # (N, C_in, H, W)
        kernel_np = np.zeros((N, C_in, KH, KW), dtype=np.float32)  # (N, C_in, KH, KW)
        pw_filter_np = np.zeros((N, C_in, C_out), dtype=np.float32)  # (N, C_in, C_out)
        bias_np = np.zeros((N, C_in), dtype=np.float32)  # (N, C_in)

        # settings
        kernel_np[0, 0, 1, 1] = 1
        kernel_np[0, 1, 1, 1] = 2
        pw_filter_np[0, 0, 0] = 3
        pw_filter_np[0, 1, 0] = 2
        pw_filter_np[0, 0, 1] = 4
        pw_filter_np[0, 0, 2] = 5

        input.set(hl.Buffer(input_np))
        kernel.set(hl.Buffer(kernel_np))
        pw_filter.set(hl.Buffer(pw_filter_np))
        bias.set(hl.Buffer(bias_np))

        dw_out = dw.realize([8, 8, 2, 2])
        out = sep_conv.realize([8, 8, 3, 2])  # (W, H, C_out)
        print(np.array(dw_out))
        print("=" * 40)
        print(np.array(out))

    aot_compile = True
    if aot_compile:
        sep_conv.compile_to(
            {
                hl.OutputFileType.object: "sep_conv/sep_conv.o",
                hl.OutputFileType.python_extension: "sep_conv/sep_conv.py.cpp",
            },
            [input, kernel, pw_filter, bias],
            "sep_conv",
        )


if __name__ == "__main__":
    main()
