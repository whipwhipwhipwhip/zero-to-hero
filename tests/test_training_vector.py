import numpy as np
import pytest

from zero_to_hero.lecture_01_micrograd.micrograd_vector.engine import Value
from zero_to_hero.lecture_01_micrograd.micrograd_vector.nn import MLP
from zero_to_hero.lecture_01_micrograd.micrograd_vector.training import (
    adam,
    adamW,
    ada_grad,
    momentum,
    rms_prop,
    sgd,
    train,
)


def make_data():
    x_data = np.array([[0.5, -0.3], [1.0, 0.2], [-0.4, 0.7]])
    # (N, 1) to match the MLP output - a (N,) target would broadcast to (N, N)
    y_data = np.array([[1.0], [-1.0], [0.3]])
    return x_data, y_data


def mse_loss(x_data, y_data, nn):
    return ((nn(Value(x_data)) - Value(y_data)) ** 2).mean()


OPTIMIZER_CASES = [
    ("sgd", sgd, {"learning_rate": 0.05}),
    ("momentum", momentum, {"alpha": 0.05, "beta": 0.9}),
    ("ada_grad", ada_grad, {"alpha": 0.5}),
    ("rms_prop", rms_prop, {"alpha": 0.1, "beta": 0.9}),
    ("adam", adam, {"alpha": 0.1, "beta": 0.9, "beta_var": 0.999}),
    ("adamW", adamW, {"alpha": 0.1, "beta": 0.9, "beta_var": 0.999, "grad_decay": 0.01}),
]


def test_mlp_output_shape_and_parameters():
    np.random.seed(0)
    nn = MLP(2, [4, 3, 1])
    x_data, _ = make_data()

    assert nn(Value(x_data)).shape == (3, 1)
    assert [p.shape for p in nn.parameters()] == [(2, 4), (4,), (4, 3), (3,), (3, 1), (1,)]


def test_zero_grad_resets_to_arrays():
    np.random.seed(0)
    nn = MLP(2, [4, 1])
    x_data, y_data = make_data()

    mse_loss(x_data, y_data, nn).backward()
    nn.zero_grad()

    for p in nn.parameters():
        assert p.grad.shape == p.data.shape
        assert not p.grad.any()


@pytest.mark.parametrize(
    "optimizer_fn, opt_params",
    [(fn, params) for _, fn, params in OPTIMIZER_CASES],
    ids=[name for name, _, _ in OPTIMIZER_CASES],
)
def test_optimizer_reduces_loss(optimizer_fn, opt_params):
    np.random.seed(42)
    nn = MLP(2, [4, 1])
    x_data, y_data = make_data()

    initial_loss = mse_loss(x_data, y_data, nn).data
    optimizer_fn(x_data, y_data, nn, 50, mse_loss, opt_params)
    final_loss = mse_loss(x_data, y_data, nn).data

    assert final_loss < initial_loss


@pytest.mark.parametrize(
    "training_method, opt_params",
    [(name, params) for name, _, params in OPTIMIZER_CASES if name != "sgd"]
    + [("SGD", {"learning_rate": 0.05})],
    ids=[name for name, _, _ in OPTIMIZER_CASES if name != "sgd"] + ["SGD"],
)
def test_train_dispatches_to_each_method(training_method, opt_params):
    np.random.seed(0)
    x_data, y_data = make_data()

    trained = train(
        x_data, y_data, nin=2, nouts=[4, 1], loss_function=mse_loss,
        opt_params=opt_params, epochs=5, training_method=training_method,
    )

    assert isinstance(trained, MLP)


def test_train_rejects_unsupported_method():
    x_data, y_data = make_data()
    with pytest.raises(AssertionError, match="SGD, momentum"):
        train(
            x_data, y_data, nin=2, nouts=[4, 1], loss_function=mse_loss,
            opt_params={}, epochs=1, training_method="not_a_real_method",
        )
