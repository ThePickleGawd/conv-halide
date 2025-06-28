"""
Runs a depthwise separable convolutions in halide
Note: np and halide have dimensions flipped
"""

import halide as hl
import numpy as np

dw = hl.Func("dw")
pw = hl.Func("pw")
x, y, c, n = hl.Var("x"), hl.Var("y"), hl.Var("c"), hl.Var("n")

input = hl.ImageParam(hl.Float(32), 4, "in")  # (W, H, C_in, N)
kernel = hl.ImageParam(hl.Float(32), 4, "k")  # (KW, KH, C_in, N)
pw_filter = hl.ImageParam(hl.Float(32), 3, "pw_filter")  # (C_out, C_in, N)

bounded = hl.BoundaryConditions.repeat_edge(input)
r = hl.RDom([(0, kernel.dim(0).extent()), (0, kernel.dim(1).extent())])  # KW, KH

# dw_conv
dw[x, y, c, n] = hl.Expr(0.0)
dw[x, y, c, n] += bounded[x - r.x + 1, y - r.y + 1, c, n] * kernel[r.x, r.y, c, n]

# pw_conv
rc = hl.RDom([(0, input.dim(2).extent())])
pw[x, y, c, n] = hl.Expr(0.0)  # TODO: Bias
pw[x, y, c, n] += dw[x, y, rc, n] * pw_filter[c, rc, n]

sep_conv = pw


# Run

input_np = np.ones((2, 2, 8, 8), dtype=np.float32)  # (N, C_in, KH, KW)
kernel_np = np.zeros((2, 2, 3, 3), dtype=np.float32)  # (N, C_in, KH, KW)
pw_filter_np = np.zeros((2, 2, 3), dtype=np.float32)  # (N, C_in, C_out)

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

dw_out = dw.realize([8, 8, 2, 2])
out = sep_conv.realize([8, 8, 3, 2])  # (W, H, C_out)
print(np.array(dw_out))
print("=" * 40)
print(np.array(out))
