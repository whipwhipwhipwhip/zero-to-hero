def forward_backward(x_data, y_data, loss_function, nn):

    # zero grad
    for p in nn.parameters():
        p.grad = 0.0

    # forward pass - what is the current loss?
    loss = loss_function(x_data, y_data, nn)

    # backward pass - get all grads going right to left - remember the gradient msut be flushed back to zero (this is .zero_grad() in pytorch)

    loss.backward()

    return loss

def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad