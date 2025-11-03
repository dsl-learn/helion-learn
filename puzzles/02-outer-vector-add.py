# Puzzle 2: Outer Vector Add
# https://helionlang.com/helion_puzzles.html#puzzle-2-outer-vector-add

import helion
import helion.language as hl
import torch
from torch import Tensor

from utils import test_kernel, benchmark_kernel, compare_implementations

def broadcast_add_spec(x: Tensor, y: Tensor) -> Tensor:
    return x[None, :] + y[:, None]

# ---- ✨ Is this the best block size? ----
@helion.kernel(config = helion.Config(block_sizes = [32, 32]))
def broadcast_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
     # ---- ✨ Your Code Here ✨----
    n0 = x.size(0)
    n1 = y.size(0)
    out = x.new_empty(n1, n0)

    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Get tiles from x and y
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        # Compute outer sum
        out[tile_i, tile_j] = y_tile[:, None] + x_tile[None, :]

    return out

# Test the kernel
x = torch.randn(1142, device="cuda")
y = torch.randn(512, device="cuda")
test_kernel(broadcast_add_kernel, broadcast_add_spec, x, y)
benchmark_kernel(broadcast_add_kernel, x, y)
compare_implementations(broadcast_add_kernel, broadcast_add_spec, x, y)
