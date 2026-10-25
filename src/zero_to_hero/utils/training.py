def forward_backward(x_data, y_data, loss_function, nn):

    # zero grad
    nn.zero_grad()

    # forward pass - what is the current loss?
    loss = loss_function(x_data, y_data, nn)

    # backward pass - get all grads going right to left - remember the gradient msut be flushed back to zero (this is .zero_grad() in pytorch)

    loss.backward()

    return loss
