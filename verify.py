import numpy as np
import torch
import torch.nn.functional as F
from sep_conv.sep_conv import sep_conv

# Define dimensions
N, H, W = 3, 8, 8
C_in, C_out = 2, 3
KH, KW = 3, 3

# Initialize numpy inputs
input_np = np.random.randn(W, H, C_in, N).astype(np.float32, order="F")
kernel_np = np.random.randn(KW, KH, C_in).astype(np.float32, order="F")
pw_filter_np = np.random.randn(C_out, C_in).astype(np.float32, order="F")
bias_np = np.random.randn(C_out).astype(np.float32, order="F")

# Output buffer
output_np = np.empty((W, H, C_out, N), dtype=np.float32, order="F")

# Call sep_conv
sep_conv(input_np, kernel_np, pw_filter_np, bias_np, output_np)
output_np = output_np.transpose()  # (N, C_out, H, W)

# --- PyTorch comparison ---
# Depthwise weights: (C_in, 1, KH, KW)
depthwise_weight = torch.from_numpy(kernel_np.transpose(2, 1, 0)).unsqueeze(1)

# Pointwise weights: (C_out, C_in, 1, 1)
pointwise_weight = torch.from_numpy(pw_filter_np).unsqueeze(-1).unsqueeze(-1)

bias_torch = torch.from_numpy(bias_np)

# Convert input: (W, H, C_in, N) -> (N, C_in, H, W)
input_torch = torch.from_numpy(input_np.transpose(3, 2, 1, 0))

# Depthwise convolution
depthwise_out = F.conv2d(
    input_torch, depthwise_weight, bias=None, groups=C_in, padding=KH // 2
)

# Pointwise convolution
torch_output = (
    F.conv2d(depthwise_out, pointwise_weight, bias=bias_torch).detach().numpy()
)


# --- Comparison ---
diff = np.abs(output_np - torch_output)
print("Max absolute difference:", diff.max())

# Count elements that differ beyond tolerance
threshold = 1e-5
num_differing = np.count_nonzero(diff > threshold)
total_elements = diff.size

print(
    f"Differing elements count (>{threshold}): {num_differing} out of {total_elements}"
)
print(f"Fraction differing: {num_differing / total_elements:.6f}")
