import numpy as np

from zero_to_hero.lecture_01_micrograd.micrograd_scalar.nn import MLP
from zero_to_hero.utils.training import forward_backward

def train(x, y, nin: int, nouts: list[int], loss_function,
         opt_params, epochs: int = 20, training_method: str = "SGD"):
  
    supported_training = ["SGD", "momentum", "ada_grad", "rms_prop", "adam", "adamW"]
    assert training_method in supported_training, f"Only training supported currently is {', '.join(supported_training)}"
    network = MLP(nin, nouts)

    match training_method:
        case "SGD":
            trained = sgd(x, y, network, epochs, loss_function, opt_params)
        case "momentum":
            trained = momentum(x, y, network, epochs, loss_function, opt_params)
        case "ada_grad":
            trained = ada_grad(x, y, network, epochs, loss_function, opt_params)
        case "rms_prop":
            trained = rms_prop(x, y, network, epochs, loss_function, opt_params)
        case "adam":
            trained = adam(x, y, network, epochs, loss_function, opt_params)
        case "adamW":
            trained = adamW(x, y, network, epochs, loss_function, opt_params)
        case _:
            raise ValueError(f"Only training supported currently is {', '.join(supported_training)}")
    
    return trained

def sgd(x_data, y_data, nn, epochs, loss_function, opt_params):

    gamma = opt_params["learning_rate"]
    for _ in range(epochs):
    
        loss = forward_backward(x_data, y_data, loss_function, nn)
        
        # update - gradient descent
        for p in nn.parameters():
            p.data += -gamma * p.grad
        
    return nn

def momentum(x_data, y_data, nn, epochs, loss_function, opt_params):

    alpha = opt_params["alpha"]
    beta = opt_params["beta"]
    for _ in range(epochs):

        loss = forward_backward(x_data, y_data, loss_function, nn)
        
        # update - momentum. Beta large, 0.9 or so.
        for p in nn.parameters():
            p.vel = beta * p.vel + (1 - beta) * p.grad
            p.data += -alpha * p.vel

    return nn

def ada_grad(x_data, y_data, nn, epochs, loss_function, opt_params):

    alpha = opt_params["alpha"]
    ep = 1e-6
    for _ in range(epochs):

        loss = forward_backward(x_data, y_data, loss_function, nn)
        
        # update - adagrad
        for p in nn.parameters():
            p.vel = p.vel + p.grad ** 2
            p.data += -alpha / (np.sqrt(p.vel) + ep) * p.grad
    
    return nn

def rms_prop(x_data, y_data, nn, epochs, loss_function, opt_params):

    alpha = opt_params["alpha"]
    beta = opt_params["beta"]
    ep = 1e-6
    for _ in range(epochs):

        loss = forward_backward(x_data, y_data, loss_function, nn)
        
        # update - rms_prop, ada_grad with EMA on vel
        for p in nn.parameters():
            p.vel = beta * p.vel + (1 - beta) * p.grad ** 2
            p.data += -alpha / (np.sqrt(p.vel) + ep) * p.grad
    
    return nn

def adam(x_data, y_data, nn, epochs, loss_function, opt_params):

    alpha = opt_params["alpha"]
    beta_vel = opt_params["beta"]
    beta_var = opt_params["beta_var"]
    ep = 1e-7
    for k in range(epochs):

        loss = forward_backward(x_data, y_data, loss_function, nn)
        
        # update - adam
        for p in nn.parameters():
            p.vel = beta_vel * p.vel + (1 - beta_vel) * p.grad
            correct_vel = p.vel / (1 - (beta_vel ** (k+1)))

            p.var = beta_var * p.var + (1 - beta_var) * (p.grad ** 2)
            correct_var = p.var / (1 - (beta_var ** (k+1)))

            p.data += -alpha * correct_vel / (np.sqrt(correct_var) + ep)

    return nn

def adamW(x_data, y_data, nn, epochs, loss_function, opt_params):

    alpha = opt_params["alpha"]
    beta_vel = opt_params["beta"]
    beta_var = opt_params["beta_var"]
    grad_decay = opt_params["grad_decay"]
    ep = 1e-7
    for k in range(epochs):

        loss = forward_backward(x_data, y_data, loss_function, nn)
        
        # update - adamW
        for p in nn.parameters():
            p.data *= (1 - alpha * grad_decay)

            p.vel = beta_vel * p.vel + (1 - beta_vel) * p.grad
            correct_vel = p.vel / (1 - (beta_vel ** (k+1)))

            p.var = beta_var * p.var + (1 - beta_var) * (p.grad ** 2)
            correct_var = p.var / (1 - (beta_var ** (k+1)))

            p.data += -alpha * correct_vel / (np.sqrt(correct_var) + ep)

    return nn