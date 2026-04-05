"""
import numpy as np
import pandas as pd

# ================= DATA LOAD =================
df = pd.read_csv("data/iris.data", header=None)

df.columns = ["f1","f2","f3","f4","label"]

# label convert
df["label"] = df["label"].map({
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
})

X = df.iloc[:,0:4].values
y = df.iloc[:,4].to_numpy(dtype=np.int64)

# one-hot
y = np.eye(3)[y]

# normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ================= MODEL =================
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp/exp.sum(axis=1,keepdims=True)

# weights
W1 = np.random.randn(4,8)*0.01
W2 = np.random.randn(8,6)*0.01
W3 = np.random.randn(6,3)*0.01

# ================= TRAIN =================
for epoch in range(500):

    # forward
    a1 = sigmoid(X @ W1)
    a2 = sigmoid(a1 @ W2)
    a3 = softmax(a2 @ W3)

    # loss
    loss = -np.mean(y*np.log(a3+1e-8))

    # backward
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

    if epoch%50==0:
        print("Epoch:",epoch,"Loss:",loss)

# ================= RESULT =================
# final forward pass for prediction
a1 = sigmoid(X @ W1)
a2 = sigmoid(a1 @ W2)
a3 = softmax(a2 @ W3)

pred = np.argmax(a3,axis=1)
true = np.argmax(y,axis=1)

accuracy = np.mean(pred==true)
print("Accuracy:",accuracy)
"""
from scr.data import load_data
from scr.train import train
from scr.evaluate import evaluate
import matplotlib.pyplot as plt

# load
X_train,X_test,y_train,y_test = load_data()

# train
W1,W2,W3,losses = train(X_train,y_train)

# evaluate
evaluate(X_test,y_test,W1,W2,W3)

# graph
plt.plot(losses)
plt.title("Loss Curve")
plt.show()

from sklearn.neural_network import MLPClassifier
import numpy as np

model = MLPClassifier(hidden_layer_sizes=(8,6))
model.fit(X_train,np.argmax(y_train,axis=1))

print("Sklearn Accuracy:",model.score(X_test,np.argmax(y_test,axis=1)))