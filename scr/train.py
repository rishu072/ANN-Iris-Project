import numpy as np
from scr.model import *

def train(X,y):
    W1,W2,W3 = init_weights()
    losses = []

    for epoch in range(500):

        a1,a2,a3 = forward(X,W1,W2,W3)

        loss = -np.mean(y*np.log(a3+1e-8))
        losses.append(loss)

        # backpropagation
        dz3 = a3 - y
        dW3 = a2.T @ dz3

        dz2 = (dz3 @ W3.T)*(a2*(1-a2))
        dW2 = a1.T @ dz2

        dz1 = (dz2 @ W2.T)*(a1*(1-a1))
        dW1 = X.T @ dz1

        # update
        W1 -= 0.01*dW1
        W2 -= 0.01*dW2
        W3 -= 0.01*dW3

        if epoch%100==0:
            print("Epoch:",epoch,"Loss:",loss)

    return W1,W2,W3,losses