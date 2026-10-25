import numpy as np


def _unbroadcast(grad, shape):
    """ sum grad down to shape, undoing any numpy broadcasting from the forward pass """
    # sum away leading dims that broadcasting added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # sum over dims that were size 1 and got stretched
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Value:
    """ stores an n-dimensional array and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.vel = np.zeros_like(self.data)
        self.var = np.zeros_like(self.data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            # for 2D: dL/dA = dL/dout @ B^T, dL/dB = A^T @ dL/dout
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            # put the summed-away axes back so the grad broadcasts to self's shape
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        n = float(n) # numpy ints can't be raised to negative powers in __truediv__
        return self.sum(axis=axis, keepdims=keepdims) / n

    @property
    def T(self):
        out = Value(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    def reshape(self, *shape):
        out = Value(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        op = np.tanh(self.data)
        out = Value(op, (self,), "tanh")

        def _backward():
            self.grad += (1 - op ** 2) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        op = 1 / (1 + np.exp(-self.data))
        out = Value(op, (self,), "sigmoid")

        def _backward():
            self.grad += op * (1 - op) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        op = np.exp(self.data)
        out = Value(op, (self,), "exp")

        def _backward():
            self.grad += op * out.grad
        out._backward = _backward

        return out

    def log(self):
        op = np.log(self.data)
        out = Value(op, (self,), "log")

        def _backward():
            self.grad +=  (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __rmatmul__(self, other): # other @ self
        return Value(other) @ self

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


def softmax(value, axis=-1):
    """ softmax of a Value along axis (default: last, i.e. per row for a (batch, classes) input) """
    # subtracting the max is for numerical stability; it's a constant so no grad flows through it
    max_val = value.data.max(axis=axis, keepdims=True)
    exps = (value - max_val).exp()
    return exps / exps.sum(axis=axis, keepdims=True)
