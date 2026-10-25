import pytest
import torch
import torch.nn.functional as F

from zero_to_hero.part_03_backprop.activations import gelu, linear, relu
from zero_to_hero.part_03_backprop.normalisation import EP, layernorm, softmax

C = 5
SHAPES = [(C,), (4, C), (2, 3, C)]
A, B = 2.0, 3.0


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def rand(shape):
    return torch.randn(shape, dtype=torch.double, requires_grad=True)


# single-input functions: name, our function, torch reference
CASES = [
    ("linear", lambda x: linear(x, A, B), lambda x: A * x + B),
    ("relu", relu, F.relu),
    ("gelu", gelu, F.gelu),
    ("softmax", softmax, lambda x: F.softmax(x, dim=-1)),
]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("fn, ref", [(f, r) for _, f, r in CASES], ids=[n for n, _, _ in CASES])
def test_matches_torch(fn, ref, shape):
    x = rand(shape)
    torch.testing.assert_close(fn(x), ref(x), atol=1e-12, rtol=0)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("fn", [f for _, f, _ in CASES], ids=[n for n, _, _ in CASES])
def test_gradcheck(fn, shape):
    assert torch.autograd.gradcheck(fn, (rand(shape),), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_layernorm_matches_torch(shape):
    x, w, b = rand(shape), rand(C), rand(C)
    torch.testing.assert_close(layernorm(x, w, b), F.layer_norm(x, (C,), w, b, EP), atol=1e-12, rtol=0)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_layernorm_gradcheck(shape):
    assert torch.autograd.gradcheck(layernorm, (rand(shape), rand(C), rand(C)), eps=1e-6, atol=1e-4)


def test_linear_rejects_trainable_constants():
    with pytest.raises(AssertionError, match="b must be a constant"):
        linear(rand(C), A, rand(C))
