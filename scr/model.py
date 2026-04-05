import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp/exp.sum(axis=1,keepdims=True)

def init_weights():
    W1 = np.random.randn(4,8)*0.01
    W2 = np.random.randn(8,6)*0.01
    W3 = np.random.randn(6,3)*0.01
    return W1,W2,W3

def forward(X,W1,W2,W3):
    a1 = sigmoid(X @ W1)
    a2 = sigmoid(a1 @ W2)
    a3 = softmax(a2 @ W3)
    return a1,a2,a3