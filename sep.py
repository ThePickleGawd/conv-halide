import halide as hl
import numpy as np

dw = hl.Func("dw")
pw = hl.Func("pw")
x, y, z = hl.Var(), hl.Var(), hl.Var()

input = hl.ImageParam(hl.Float(32), 3, "in")  # (W, H, C_in)
kernel = hl.ImageParam(hl.Float(32), 3, "k")  # (KW, KH, C_in)
pw_filter = hl.ImageParam(hl.Float(32), 2, "pw_filter")  # (C_out, C_in)

bounded = hl.BoundaryConditions.repeat_edge(input)
r = hl.RDom([(0, kernel.dim(0).extent()), (0, kernel.dim(1).extent())])  # KW, KH

# dw_conv
dw[x, y, z] = hl.Expr(0.0)
dw[x, y, z] += bounded[x - r.x + 1, y - r.y + 1, z] * kernel[r.x, r.y, z]

# pw_conv
rc = hl.RDom([(0, input.dim(2).extent())])
pw[x, y, z] = hl.Expr(0.0)  # TODO: Bias
pw[x, y, z] += dw[x, y, rc] * pw_filter[z, rc]

sep_conv = pw


# Run

input_np = np.ones((2, 8, 8), dtype=np.float32)  # Flipped ??
kernel_np = np.zeros((2, 3, 3), dtype=np.float32)
pw_filter_np = np.zeros((2, 3), dtype=np.float32)  # Flipped ?? (C_in, C_out)

# settings
kernel_np[0, 1, 1] = 1
kernel_np[1, 1, 1] = 2
pw_filter_np[0, 0] = 3
pw_filter_np[1, 0] = 2
pw_filter_np[0, 1] = 4
pw_filter_np[0, 2] = 5

input.set(hl.Buffer(input_np))
kernel.set(hl.Buffer(kernel_np))
pw_filter.set(hl.Buffer(pw_filter_np))

dw_out = dw.realize([8, 8, 2])
out = sep_conv.realize([8, 8, 3])  # (W, H, C_out)
print(np.array(dw_out))
print("=" * 40)
print(np.array(out))
