import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
    
def relu(Z):
    return np.maximum(0, Z)

def tanh(Z):
    return np.tanh(Z)

def sigmoid_backward(A):
    return sigmoid(A) * (1 - sigmoid(A))

def relu_backward(A):
    A[A <= 0] = 0
    A[A > 0] = 1
    return A

def tanh_backward(A):
    return 1 - np.power(tanh(A), 2)