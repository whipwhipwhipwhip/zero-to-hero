## Part 1 - Micrograd

Material from [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0). The lecture builds a tiny scalar autograd engine (a `Value` class that records the operations applied to it, then backpropagates through them in topological order) and a small MLP library on top of it.

### What's here

- [micrograd_scalar/](micrograd_scalar) - the engine from the lecture, where every `Value` wraps a single float. On top of the lecture's ops it adds `sigmoid`, `exp`, `log` and a `softmax` helper.
- [micrograd_vector/](micrograd_vector) - my extension of the engine to numpy arrays. A `Value` holds an n-dimensional array, and the engine supports `@`, `sum`, `mean`, `reshape` and transpose. Gradients are summed back down to the input's shape wherever numpy broadcast it in the forward pass. Because the graph has one node per array op rather than one per scalar op, it's orders of magnitude faster (see the scalar vs vector notebook).
- `training.py` (identical in both packages) - a `train` function plus several optimisers: SGD, momentum, AdaGrad, RMSProp, Adam and AdamW. Optimiser state (velocity / squared-gradient averages) is stored on each `Value` alongside its gradient.

### Notebooks

In [notebooks/01_micrograd](../../../notebooks/01_micrograd):

- `lecture_01.ipynb` - follow-along notes for the lecture.
- `lecture_01_pytorch.ipynb` - the same steps redone with PyTorch tensors and autograd, for comparison.
- `exercises_01.ipynb` - the lecture's exercises: analytical vs numerical gradients, and adding the ops needed to backprop through softmax.
- `scalar_vs_vector.ipynb` - builds the same MLP in both engines and checks that the losses, gradients and training curves match. It then times a larger batch: on a 200-sample batch the vector engine was ~1000x faster.

### Tests

From the repo root:

```bash
uv run pytest tests/test_derivatives.py tests/test_derivatives_vector.py tests/test_training.py tests/test_training_vector.py
```

The derivative tests compare each op's backward pass against a numerical gradient. The training tests check that every optimiser reduces the loss on a small regression problem.
