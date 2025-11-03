# Puzzle 1: Constant Add
# https://helionlang.com/helion_puzzles.html#puzzle-1-constant-add

import helion
import helion.language as hl
import torch
from torch import Tensor

from utils import test_kernel, benchmark_kernel, compare_implementations


def add_spec(x: Tensor) -> Tensor:
    """This is the spec that you should implement."""
    return x + 10.

# ---- ✨ Is this the best block size? ----
@helion.kernel(config = helion.Config(block_sizes = [32,]))
def add_kernel(x: torch.Tensor) -> torch.Tensor:
    # ---- ✨ Your Code Here ✨----
    # Set up the output buffer which you will return
    TILE_RANGE = x.size()
    out = torch.empty_like(x)
    # ---- End of Code ----

    # Use Helion to tile the computation
    for tile_n in hl.tile(TILE_RANGE):
         # ---- ✨ Your Code Here ✨----
        x_tile = x[tile_n]
        out[tile_n] = x_tile + 10.0

    return out

# Test the kernel
x = torch.randn(8192, device="cuda")
test_kernel(add_kernel, add_spec, x)
benchmark_kernel(add_kernel, x)
compare_implementations(add_kernel, add_spec, x)
