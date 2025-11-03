# Puzzle 10: Two Dimensional Convolution
# https://helionlang.com/helion_puzzles.html#puzzle-10-two-dimensional-convolution

import helion
import helion.language as hl
import torch
from torch import Tensor

from jaxtyping import Float32

from utils import test_kernel, benchmark_kernel, compare_implementations

def conv2d_spec(x: Float32[Tensor, "4 8 8"], k: Float32[Tensor, "4 4"]) -> Float32[Tensor, "4 8 8"]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
    return z

@helion.kernel()
def conv2d_kernel(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, h, w = x.size()
    kh, kw = k.size()[1:]

    # Create output tensor
    out = torch.empty_like(x)

    # Pad the input
    x_padded = torch.nn.functional.pad(x, (0, kw, 0, kh, 0, 0), value=0.0)

    # Use Helion to tile the computation
    for tile_batch in hl.tile(batch):
        # Process each output position
        for i in range(h):
            for j in range(w):
                # Extract the patch
                patch = x_padded[tile_batch, i:i+kh, j:j+kw]
                # Apply the kernel
                out[tile_batch, i, j] = (k[tile_batch] * patch).sum([1, 2])

    return out

# Test the kernel
x = torch.randn(4, 8, 8, device="cuda")
k = torch.randn(4, 4, 4, device="cuda")
test_kernel(conv2d_kernel, conv2d_spec, x, k)
