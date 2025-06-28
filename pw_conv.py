import halide as hl
import numpy as np

f = hl.Func("pointwise")
x, y, z = hl.Var(), hl.Var(), hl.Var()

input = hl.ImageParam(hl.Float(32), 3, "in")  # (W, H, C_in)
pw_filter = hl.ImageParam(hl.Float(32), 2, "pw_filter")  # (C_out, C_in)

bounded = hl.BoundaryConditions.repeat_edge(input)
r = hl.RDom([(0, input.dim(2).extent())])
f[x, y, z] = hl.Expr(0.0)
f[x, y, z] += input[x, y, r[0]] * pw_filter[z, r[0]]

input_np = np.ones((2, 8, 8), dtype=np.float32)  # Flipped ??
pw_filter_np = np.zeros((2, 3), dtype=np.float32)  # Flipped ?? (C_in, C_out)
pw_filter_np[0, 0] = 5
pw_filter_np[0, 1] = 10
pw_filter_np[1, 0] = 2

input.set(hl.Buffer(input_np))
pw_filter.set(hl.Buffer(pw_filter_np))

out = f.realize([8, 8, 3])  # (W, H, C_out)
print(np.array(out))
