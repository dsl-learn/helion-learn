import helion
import helion.language as hl
import torch


@helion.kernel(config=helion.Config(block_sizes = [1024]))  # The @helion.kernel decorator marks this function for compilation
def example_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Host code: Standard PyTorch operations
    out = torch.empty_like(x)  # Allocate output tensor

    n_elements = x.size()
    # The hl.tile loop defines the parallel execution structure
    for block_size in hl.tile(n_elements):
        # Device code: Everything inside the hl.tile loop runs on GPU
        out[block_size] = x[block_size] + y[block_size] # Simple element-wise addition expressed w/ pytorch ops

    return out  # Return the result back to the host

# Create some sample data
size = 98432
x = torch.randn(size, device="cuda")
y = torch.randn(size, device="cuda")

# Run the kernel
result = example_add(x, y)

# Verify result
expected = x + y
torch.testing.assert_close(result, expected)
print("✅ Results Match ✅")
