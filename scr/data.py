import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/iris.data", header=None)

    df.columns = ["f1","f2","f3","f4","label"]

    # label ko number me convert karna
    df["label"] = df["label"].map({
        "Iris-setosa":0,
        "Iris-versicolor":1,
        "Iris-virginica":2
    })

    X = df.iloc[:,0:4].values                    # features
    y = df.iloc[:,4].to_numpy(dtype=np.int64)   # labels

    y = np.eye(3)[y]  # one-hot encoding

    # data split
    X_train,X_temp,y_train,y_temp = train_test_split(X,y,test_size=0.3)
    X_test = X_temp
    y_test = y_temp

    # normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train-mean)/std
    X_test = (X_test-mean)/std

    return X_train,X_test,y_train,y_test