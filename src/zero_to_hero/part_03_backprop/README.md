## Part 3 - Backpropagation

Material from [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI). The lecture removes `loss.backward()` and backpropagates through the makemore MLP by hand, one tensor at a time, including cross-entropy, batchnorm and tanh.

There are no notebooks for this part. Instead, the code writes the forward and backward passes of some common layers by hand, as `torch.autograd.Function`s, and checks them against PyTorch.

### What's here

- [activations.py](activations.py) - `linear` (`a * x + b` with constant `a`, `b`), `relu` and `gelu` (the exact erf form).
- [normalisation.py](normalisation.py):
  - `softmax` - numerically stable forward pass. The backward pass uses `out * (grad - (grad * out).sum())` rather than building the full Jacobian.
  - `layernorm` - normalises over the last dimension with learnable weight and bias. The backward pass recomputes the normalised input rather than storing it, and reduces the gradient of `x` to the usual three-term formula.

Each function works on inputs with any number of leading batch dimensions.

### Tests

From the repo root:

```bash
uv run pytest tests/test_backprop.py
```

The tests compare each function's output and gradients against the PyTorch built-in, and run `torch.autograd.gradcheck` against numerical gradients.
