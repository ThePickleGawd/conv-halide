import halide as hl
import numpy as np

f = hl.Func("f")
x, y, z = hl.Var(), hl.Var(), hl.Var()

input = hl.ImageParam(hl.Float(32), 3, "in")
kernel = hl.ImageParam(hl.Float(32), 3, "k")

bounded = hl.BoundaryConditions.repeat_edge(input)
r = hl.RDom([(0, 3), (0, 3)])
f[x, y, z] = hl.Expr(0.0)
f[x, y, z] += bounded[x - r.x + 1, y - r.y + 1, z] * kernel[r.x, r.y, z]

input_np = np.ones((2, 8, 8), dtype=np.float32) # Flipped ??
kernel_np = np.zeros((2, 3, 3), dtype=np.float32)
kernel_np[0, 1, 1] = 2

input.set(hl.Buffer(input_np))
kernel.set(hl.Buffer(kernel_np))

out = f.realize([8, 8, 2])
print(np.array(out))
