# Puzzle 3: Fused Outer Multiplication
# https://helionlang.com/helion_puzzles.html#puzzle-3-fused-outer-multiplication

import helion
import helion.language as hl
import torch
from torch import Tensor

from utils import test_kernel, benchmark_kernel, compare_implementations

def mul_relu_block_spec(x: Tensor, y: Tensor) -> Tensor:
    return torch.relu(x[None, :] * y[:, None])

# ---- ✨ Is this the best block size? ----
@helion.kernel(config = helion.Config(block_sizes = [32, 32]))
def mul_relu_block_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    n0 = x.size(0)
    n1 = y.size(0)
    # Create output tensor
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)

    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Get tiles from x and y
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        # Compute outer product followed by ReLU
        out[tile_i, tile_j] = torch.relu(y_tile[:, None] * x_tile[None, :])

    return out

# Test the kernel
x = torch.randn(512, device="cuda")
y = torch.randn(512, device="cuda")
test_kernel(mul_relu_block_kernel, mul_relu_block_spec, x, y)
compare_implementations(mul_relu_block_kernel, mul_relu_block_spec, x, y)
