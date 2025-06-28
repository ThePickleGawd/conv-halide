from sep_conv.sep_conv import sep_conv
import numpy as np

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

output = np.empty((N, C_out, H, W), dtype=np.float32, order="C")

sep_conv(input_np, kernel_np, pw_filter_np, bias_np, output)
