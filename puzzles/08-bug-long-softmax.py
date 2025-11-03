# Puzzle 8: Long Softmax
# https://helionlang.com/helion_puzzles.html#puzzle-8-long-softmax

import helion
import helion.language as hl
import torch
from torch import Tensor

from jaxtyping import Float32

from utils import test_kernel, benchmark_kernel, compare_implementations

def softmax_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4 200"]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)

@helion.kernel()
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, seq_len = x.size()
    # Create output tensor
    out = torch.empty_like(x)

    # Use Helion to tile the batch dimension
    for tile_batch in hl.tile(batch):
        # First pass: find max value for each sequence
        max_vals = torch.full_like(tile_batch, float('-inf'), dtype=torch.float32)

        for tile_seq in hl.tile(seq_len):
            chunk = x[tile_batch, tile_seq]
            max_vals = torch.maximum(max_vals, torch.max(chunk, dim=1)[0])

        # Second pass: compute sum of exp(x - max)
        sum_exp = torch.zeros_like(tile_batch, dtype=torch.float32)

        for tile_seq in hl.tile(seq_len):
            chunk = x[tile_batch, tile_seq]
            exp_vals = torch.exp(chunk - max_vals[:, None])
            sum_exp += torch.sum(exp_vals, dim=1)

        # Third pass: compute softmax
        for tile_seq in hl.tile(seq_len):
            chunk = x[tile_batch, tile_seq]
            exp_vals = torch.exp(chunk - max_vals[:, None])
            out[tile_batch, tile_seq] = exp_vals / sum_exp[:, None]

    return out

# Test the kernel
x = torch.randn(4, 200, device="cuda")
test_kernel(softmax_kernel, softmax_spec, x)
