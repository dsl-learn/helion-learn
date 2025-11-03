# Puzzle 9: Simple FlashAttention
# https://helionlang.com/helion_puzzles.html#puzzle-9-simple-flashattention

import helion
import helion.language as hl
import torch
from torch import Tensor

from jaxtyping import Float32

from utils import test_kernel, benchmark_kernel, compare_implementations

def flashatt_spec(q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)

@helion.kernel()
def flashatt_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Get tensor size
    seq_len = q.size(0)
    # Create output tensor
    out = torch.empty_like(q)

    # Process each query position
    for tile_q in hl.tile(seq_len):
        q_tile = q[tile_q]

        # Initialize tracking variables for stable softmax
        max_val = torch.full_like(q_tile, float('-inf'))
        sum_exp = torch.zeros_like(q_tile)
        weighted_sum = torch.zeros_like(q_tile)

        # Process in tiles for better cache efficiency
        for tile_kv in hl.tile(seq_len):
            k_tile = k[tile_kv]
            v_tile = v[tile_kv]

            # Compute attention scores
            scores = q_tile[:, None] * k_tile[None, :]

            # Find max for numerical stability
            batch_max = torch.max(scores, dim=1)[0]
            new_max = torch.maximum(max_val, batch_max)

            # Scale old accumulations
            scale_factor = torch.exp(max_val - new_max)
            sum_exp = sum_exp * scale_factor
            weighted_sum = weighted_sum * scale_factor

            # Update with new values
            exp_scores = torch.exp(scores - new_max[:, None])
            sum_exp = sum_exp + torch.sum(exp_scores, dim=1)
            weighted_sum = weighted_sum + torch.sum(exp_scores * v_tile[None, :], dim=1)

            # Update max_val
            max_val = new_max

        # Compute final output
        out[tile_q] = weighted_sum / sum_exp

    return out

# Test the kernel
q = torch.randn(200, device="cuda")
k = torch.randn(200, device="cuda")
v = torch.randn(200, device="cuda")
test_kernel(flashatt_kernel, flashatt_spec, q, k, v)
