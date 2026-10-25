import math

import torch

class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a=1, b=0):
        # a and b are treated as constants - no grad flows to them
        assert not (torch.is_tensor(a) and a.requires_grad), "a must be a constant"
        assert not (torch.is_tensor(b) and b.requires_grad), "b must be a constant"
        ctx.a = a
        return a * x + b

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output * ctx.a
        return grad_x, None, None

class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * (x > 0).to(grad_output.dtype)
        return grad_x

class GeLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        phi = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        grad_x = grad_output * (Phi + x * phi)
        return grad_x


def linear(x, a=1, b=0):
    return Linear.apply(x,a,b)

def relu(x):
    return ReLU.apply(x)

def gelu(x):
    return GeLU.apply(x)
