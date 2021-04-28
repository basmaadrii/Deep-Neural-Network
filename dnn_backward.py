import numpy as np
from activation import sigmoid_backward, relu_backward, tanh_backward

def linear_activation_backward(A_prev, Z, W, dA, activation):
    m = dA.shape[1]
    dZ = dA * (sigmoid_backward(Z) if activation == "sigmoid" else relu_backward(Z))
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def layer_wise_backward(parameters, cache, grads, l, dA, activation):
    A_prev = cache["A" + str(l - 1)]    
    Z = cache["Z" + str(l)]
    W = parameters["W" + str(l)]

    dA_prev, dW, db = linear_activation_backward(A_prev, Z, W, dA, activation)

    grads["dW" + str(l)] = dW
    grads["db" + str(l)] = db
    return dA_prev

def L_model_backward(AL, Y, parameters, cache):
    dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    L = len(parameters) // 2
    grads = {}

    dA = layer_wise_backward(parameters, cache, grads, L, dAL, "sigmoid")

    for l in reversed(range(1, L)):
        dA = layer_wise_backward(parameters, cache, grads, l, dA, "relu")

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        dW = grads["dW" + str(l)]
        db = grads["db" + str(l)]

        W -= learning_rate * dW
        b -= learning_rate * db

        parameters["W" + str(l)] = W
        parameters["b" + str(l)] = b