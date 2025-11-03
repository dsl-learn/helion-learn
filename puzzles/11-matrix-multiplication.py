# Puzzle 11: Matrix Multiplication
# https://helionlang.com/helion_puzzles.html#puzzle-11-matrix-multiplication

import helion
import helion.language as hl
import torch
from torch import Tensor

from jaxtyping import Float32

from utils import test_kernel, benchmark_kernel, compare_implementations

def dot_spec(x: Float32[Tensor, "4 32 32"], y: Float32[Tensor, "4 32 32"]) -> Float32[Tensor, "4 32 32"]:
    return x @ y

@helion.kernel()
def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, m, k = x.size()
    _, k, n = y.size()

    # Create output tensor
    out = torch.empty([batch, m, n], dtype=x.dtype, device=x.device)

    # Use Helion to tile the computation
    for tile_batch in hl.tile(batch):
        for tile_m, tile_n in hl.tile([m, n]):
            # Initialize accumulator
            acc = hl.zeros([tile_batch, tile_m, tile_n], dtype=torch.float32)

            # Process the reduction dimension in tiles
            for tile_k in hl.tile(k):
                # Get tiles
                x_tile = x[tile_batch, tile_m, tile_k]
                y_tile = y[tile_batch, tile_k, tile_n]

                # Accumulate matrix multiplication
                acc = acc + torch.matmul(x_tile, y_tile)

            # Store result
            out[tile_batch, tile_m, tile_n] = acc

    return out

# Test the kernel
x = torch.randn(4, 32, 32, device="cuda")
y = torch.randn(4, 32, 32, device="cuda")
test_kernel(dot_kernel, dot_spec, x, y)