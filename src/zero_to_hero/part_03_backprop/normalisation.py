import torch

EP = 1e-5
class SoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # stability
        x = x - x.amax(-1, keepdim=True)
        exps = torch.exp(x)
        out = exps / exps.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # This materialises the whole matrix and is slow
        #grad_x = grad_output.T @ (torch.diag(out) - torch.outer(out, out))
        (out,) = ctx.saved_tensors
        dot = (grad_output * out).sum(-1, keepdim=True)

        return out * (grad_output - dot)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        mean = torch.mean(x, dim=-1, keepdim=True)
        rstd = torch.rsqrt(torch.var(x, unbiased=False, dim=-1, keepdim=True) + EP)
        norm = (x - mean) * rstd
        ctx.save_for_backward(x, w, mean, rstd)

        return norm * w + b

    @staticmethod
    def backward(ctx, grad_output):
        (x, w, mean, rstd) = ctx.saved_tensors
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # fold every batch dim into one: sum(dim=()) would collapse all dims
        # rather than none, which is wrong for 1D input
        C = grad_output.shape[-1]
        # bias
        db = grad_output.reshape(-1, C).sum(0)
        # weight
        dw = (grad_output * norm).reshape(-1, C).sum(0)
        # x - recall three terms from notes
        dnorm = grad_output * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db



def softmax(x):
    return SoftMax.apply(x)

def layernorm(x,w,b):
    return LayerNorm.apply(x,w,b)
