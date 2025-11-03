# Puzzle 7: Long Sum
# https://helionlang.com/helion_puzzles.html#puzzle-7-long-sum

import helion
import helion.language as hl
import torch
from torch import Tensor

from jaxtyping import Float32

from utils import test_kernel, benchmark_kernel, compare_implementations

def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, seq_len = x.size()
    # Create output tensor
    out = torch.empty(batch, dtype=x.dtype, device=x.device)

    # Use Helion to tile the batch dimension
    for tile_batch in hl.tile(batch):
        # Initialize accumulator for each batch element
        acc = torch.zeros(tile_batch, dtype=torch.float32, device=x.device)

        # Process the sequence in chunks
        for tile_seq in hl.tile(seq_len):
            # Get the current chunk
            chunk = x[tile_batch, tile_seq]
            # Accumulate sum
            acc += torch.sum(chunk, dim=1)

        # Store result
        out[tile_batch] = acc

    return out

# Test the kernel
x = torch.randn(4, 200, device="cuda")
test_kernel(sum_kernel, sum_spec, x)
