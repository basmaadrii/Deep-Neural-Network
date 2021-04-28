import numpy as np
from activation import sigmoid, relu, tanh


def initialize_parameters(layer_dims):
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    return A, Z

def layer_wise_forward(A, parameters, cache, l, activation):
    W = parameters["W" + str(l)]
    b = parameters["b" + str(l)]
    A, Z = linear_activation_forward(A, W, b, activation)
    cache["A" + str(l)] = A
    cache["Z" + str(l)] = Z
    return A

def L_model_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    cache = {}
    cache["A0"] = X

    for l in range(1, L):
        A = layer_wise_forward(A, parameters, cache, l, "relu")

    AL = layer_wise_forward(A, parameters, cache, L, "sigmoid")
    return AL, cache


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1/m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost) 
    return cost
