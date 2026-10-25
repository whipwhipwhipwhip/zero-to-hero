import numpy as np
import pytest

from ml_learning.micrograd_vector.engine import Value
from ml_learning.utils.differentiation import derivative_numerical


def assert_grad_matches_numerical(value_fn, numeric_fn, x0, atol=1e-3):
    x = Value(x0)
    out = value_fn(x)
    out.backward()

    expected = derivative_numerical(numeric_fn, x0)
    assert x.grad == pytest.approx(expected, abs=atol)


UNARY_CASES = [
    ("tanh", lambda v: v.tanh(), np.tanh, [-2.0, -0.5, 0.0, 0.5, 2.0]),
    ("sigmoid", lambda v: v.sigmoid(), lambda x: 1 / (1 + np.exp(-x)), [-3.0, -1.0, 0.0, 1.0, 3.0]),
    ("exp", lambda v: v.exp(), np.exp, [-2.0, -0.5, 0.0, 0.5, 2.0]),
    ("log", lambda v: v.log(), np.log, [0.1, 0.5, 1.0, 2.0, 5.0]),
    # avoid x == 0, relu is non-differentiable at the kink
    ("relu", lambda v: v.relu(), lambda x: max(x, 0.0), [-2.0, -0.5, 0.5, 2.0]),
]


@pytest.mark.parametrize(
    "value_fn, numeric_fn, x0",
    [
        (value_fn, numeric_fn, x0)
        for _, value_fn, numeric_fn, points in UNARY_CASES
        for x0 in points
    ],
    ids=[
        f"{name}[{x0}]"
        for name, _, _, points in UNARY_CASES
        for x0 in points
    ],
)
def test_unary_op_gradients(value_fn, numeric_fn, x0):
    assert_grad_matches_numerical(value_fn, numeric_fn, x0)


OTHER_OPERANDS = [-2.0, -0.5, 0.5, 2.0]
POWERS = [-2, -1, 0.5, 2, 3]


@pytest.mark.parametrize("other", OTHER_OPERANDS)
@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.5, 2.0])
def test_add_gradient(x0, other):
    assert_grad_matches_numerical(lambda v: v + other, lambda x: x + other, x0)


@pytest.mark.parametrize("other", OTHER_OPERANDS)
@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.5, 2.0])
def test_mul_gradient(x0, other):
    assert_grad_matches_numerical(lambda v: v * other, lambda x: x * other, x0)


@pytest.mark.parametrize("power", POWERS)
@pytest.mark.parametrize("x0", [0.5, 1.0, 2.0, 3.0])
def test_pow_gradient(x0, power):
    assert_grad_matches_numerical(lambda v: v**power, lambda x: x**power, x0)


@pytest.mark.parametrize("other", OTHER_OPERANDS)
@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.5, 2.0])
def test_sub_gradient(x0, other):
    assert_grad_matches_numerical(lambda v: v - other, lambda x: x - other, x0)


@pytest.mark.parametrize("other", OTHER_OPERANDS)
@pytest.mark.parametrize("x0", [-2.0, -0.5, 0.5, 2.0])
def test_truediv_gradient(x0, other):
    assert_grad_matches_numerical(lambda v: v / other, lambda x: x / other, x0)
