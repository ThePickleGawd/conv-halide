"""
sep_conv:
    input_feature_map: ndarray of shape (W, H, C_in, N)
    kernel:            ndarray of shape (KW, KH, C_in, N)
    pw_filter:         ndarray of shape (C_out, C_in, N)
    bias:              ndarray of shape (C_out, N)
    output:            ndarray of shape (W, H, C_out, N)
"""

from sep_conv.sep_conv import sep_conv
import numpy as np

N, H, W = 2, 8, 8
C_in, C_out = 2, 3
KH, KW = 3, 3

input_np = np.ones((W, H, C_in, N), dtype=np.float32, order="F")  # (W, H, C_in, N)
kernel_np = np.zeros(
    (KW, KH, C_in, N), dtype=np.float32, order="F"
)  # (KW, KH, C_in, N)
pw_filter_np = np.zeros(
    (C_out, C_in, N), dtype=np.float32, order="F"
)  # (C_out, C_in, N)
bias_np = np.zeros((C_out, N), dtype=np.float32, order="F")  # (C_out, N)

# Example weights: set something for batch 0
kernel_np[1, 1, 0, 0] = 1
kernel_np[1, 1, 1, 0] = 2

pw_filter_np[0, 0, 0] = 3
pw_filter_np[0, 1, 0] = 2
pw_filter_np[1, 0, 0] = 4
pw_filter_np[2, 0, 0] = 5

output = np.empty((W, H, C_out, N), dtype=np.float32, order="F")

sep_conv(input_np, kernel_np, pw_filter_np, bias_np, output)
output = output.transpose()

print(output)  # [[7...] [4...] [5...]], [[0...], ...]
print(output.shape)  # (2, 3, 8, 8)
