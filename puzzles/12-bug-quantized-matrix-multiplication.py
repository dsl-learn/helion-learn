# Puzzle 12: Quantized Matrix Multiplication
# https://helionlang.com/helion_puzzles.html#puzzle-12-quantized-matrix-multiplication

import helion
import helion.language as hl
import torch
from torch import Tensor

from jaxtyping import Float32, Int32

from utils import test_kernel, benchmark_kernel, compare_implementations

FPINT = 32 // 4
GROUP = 8

def quant_dot_spec(scale: Float32[Tensor, "32 8"],
                   offset: Int32[Tensor, "32"],
                   weight: Int32[Tensor, "32 8"],
                   activation: Float32[Tensor, "64 32"]) -> Float32[Tensor, "32 32"]:
    offset = offset.view(32, 1)
    def extract(x):
        over = torch.arange(8, device=x.device) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask
    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    return (scale * (extract(weight).view(-1, 64) - offset)) @ activation

@helion.kernel()
def quant_dot_kernel(scale: torch.Tensor, offset: torch.Tensor, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    n_out, n_groups = scale.size()
    mid, n_in = activation.size()

    # Create output tensor
    out = torch.empty([n_out, n_in], dtype=scale.dtype, device=scale.device)

    # Helper function to extract 4-bit values
    def extract_4bit(x, bit_positions):
        mask = 2**4 - 1
        shifted = x[..., None] >> (bit_positions * 4)
        return shifted & mask

    # Bit positions for extraction
    bit_positions = torch.arange(8, device=scale.device)

    # Use Helion to tile the computation
    for tile_out in hl.tile(n_out):
        for tile_in in hl.tile(n_in):
            # Initialize accumulator
            acc = hl.zeros([tile_out, tile_in], dtype=torch.float32)

            # Get the offset values for this tile
            offset_tile = offset[tile_out]
            # Extract 4-bit values from offsets
            offset_extracted = extract_4bit(offset_tile, bit_positions)

            # Process in chunks across the middle dimension
            for group_idx in range(n_groups):
                # Get scale for this group
                scale_group = scale[tile_out, group_idx]

                # Get weights for this group
                weight_group = weight[tile_out, group_idx]

                # Extract 4-bit values from weights
                weight_extracted = extract_4bit(weight_group, bit_positions)

                # Compute dequantized weights: scale * (weight - offset)
                offset_group = offset_extracted[:, group_idx:group_idx+1]  # Shape: [tile_out, 1, 8]
                dequant_weights = scale_group[:, None, None] * (weight_extracted - offset_group)

                # Reshape dequantized weights for matrix multiplication
                dequant_weights = dequant_weights.reshape(tile_out.size(0), 8)

                # Get activations for this group
                acts_idx = group_idx * 8 + torch.arange(8, device=scale.device)
                act_group = activation[acts_idx][:, tile_in]

                # Accumulate to result
                acc = acc + torch.matmul(dequant_weights, act_group)

            # Store result
            out[tile_out, tile_in] = acc

    return out

# Test the kernel with smaller inputs for quicker testing
scale = torch.randn(32, 8, device="cuda")
offset = torch.randint(-10, 10, (32,), device="cuda")
weight = torch.randint(0, 16, (32, 8), device="cuda", dtype=torch.int32)
activation = torch.randn(64, 32, device="cuda")
test_kernel(quant_dot_kernel, quant_dot_spec, scale, offset, weight, activation)
