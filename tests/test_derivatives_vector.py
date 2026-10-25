import numpy as np
import pytest

from zero_to_hero.part_01_micrograd.micrograd_vector.engine import Value, softmax


def numerical_grads(f, arrays, h=1e-6):
    """ central-difference gradient of f(*arrays).sum() w.r.t. each array """
    grads = []
    for i, a in enumerate(arrays):
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            plus = [x.copy() for x in arrays]
            minus = [x.copy() for x in arrays]
            plus[i][idx] += h
            minus[i][idx] -= h
            f_plus = f(*map(Value, plus)).data.sum()
            f_minus = f(*map(Value, minus)).data.sum()
            g[idx] = (f_plus - f_minus) / (2 * h)
        grads.append(g)
    return grads


def assert_grads_match_numerical(f, *shapes, atol=1e-5):
    rng = np.random.default_rng(0)
    # keep inputs positive and away from 0 so log / relu kinks don't bite
    arrays = [rng.uniform(0.5, 1.5, s) for s in shapes]
    values = [Value(a) for a in arrays]

    f(*values).sum().backward()

    for v, expected in zip(values, numerical_grads(f, arrays)):
        assert v.grad.shape == v.data.shape
        np.testing.assert_allclose(v.grad, expected, atol=atol)


CASES = [
    ("add", lambda x, y: x + y, [(4, 3), (4, 3)]),
    ("add_broadcast_row", lambda x, b: x + b, [(4, 3), (3,)]),
    ("add_broadcast_col", lambda x, b: x + b, [(4, 3), (4, 1)]),
    ("mul_broadcast", lambda x, b: x * b, [(4, 3), (3,)]),
    ("sub_broadcast", lambda x, b: x - b, [(4, 3), (1, 3)]),
    ("div_broadcast", lambda x, y: x / y, [(4, 3), (3,)]),
    ("pow", lambda x: x ** 3, [(4, 3)]),
    ("matmul", lambda a, b: a @ b, [(4, 3), (3, 2)]),
    ("sum_all", lambda x: x.sum() * x, [(4, 3)]),
    ("sum_axis0", lambda x: x.sum(axis=0) * Value(np.arange(3.0)), [(4, 3)]),
    ("sum_axis1_keepdims", lambda x: x.sum(axis=1, keepdims=True) * x, [(4, 3)]),
    ("mean_axis1", lambda x: x.mean(axis=1) ** 2, [(4, 3)]),
    ("transpose", lambda x, w: x.T @ w, [(4, 3), (4, 2)]),
    ("reshape", lambda x: x.reshape(2, 6) ** 2, [(4, 3)]),
    ("relu", lambda x: (x - 1).relu(), [(4, 3)]),
    ("tanh", lambda x: x.tanh(), [(4, 3)]),
    ("sigmoid", lambda x: x.sigmoid(), [(4, 3)]),
    ("exp", lambda x: x.exp(), [(4, 3)]),
    ("log", lambda x: x.log(), [(4, 3)]),
    # weight the outputs, otherwise each softmax row sums to 1 and every grad is 0
    ("softmax", lambda x: softmax(x) * Value(np.arange(12.0).reshape(4, 3)), [(4, 3)]),
    ("layer", lambda x, W, b: ((x @ W + b).tanh() @ W.T).mean(), [(5, 3), (3, 4), (4,)]),
]


@pytest.mark.parametrize("f, shapes", [(f, s) for _, f, s in CASES], ids=[n for n, _, _ in CASES])
def test_op_gradients(f, shapes):
    assert_grads_match_numerical(f, *shapes)


def test_softmax_rows_sum_to_one():
    out = softmax(Value(np.random.default_rng(0).normal(size=(4, 3))))
    np.testing.assert_allclose(out.data.sum(axis=1), np.ones(4))


def test_grads_accumulate_when_value_reused():
    x = Value(np.array([1.0, 2.0]))
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)
