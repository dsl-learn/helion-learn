# Puzzle 4: Fused Outer Multiplication - Backwards
# https://helionlang.com/helion_puzzles.html#puzzle-4-fused-outer-multiplication-backwards

import helion
import helion.language as hl
import torch
from torch import Tensor

from utils import test_kernel, benchmark_kernel, compare_implementations

def mul_relu_block_back_spec(x: Tensor, y: Tensor, dz: Tensor) -> Tensor:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
    return grad_x

@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def mul_relu_block_back_kernel(
    x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor
) -> torch.Tensor:
    # Get tensor sizes
    n0 = x.size(1)
    n1 = x.size(0)
    # Create output tensor for gradients
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)

    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Get input tiles
        x_tile = x[tile_i, tile_j]
        y_tile = y[tile_i]
        dz_tile = dz[tile_i, tile_j]

        # Compute gradients for ReLU * multiplication backward
        # For ReLU, gradient is 1 where input > 0, 0 otherwise
        relu_mask = (x_tile * y_tile[:, None]) > 0
        # Chain rule: dx = dz * relu_grad * y
        dx[tile_i, tile_j] = dz_tile * relu_mask * y_tile[:, None]

    return dx

# Test the kernel
x = torch.randn(512, 1024, device="cuda")
y = torch.randn(512, device="cuda")
dz = torch.randn(512, 1024, device="cuda")
test_kernel(mul_relu_block_back_kernel, mul_relu_block_back_spec, x, y, dz)
