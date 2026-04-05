import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from scr.model import forward

def evaluate(X,y,W1,W2,W3):

    _,_,a3 = forward(X,W1,W2,W3)

    pred = np.argmax(a3,axis=1)
    true = np.argmax(y,axis=1)

    print("Accuracy:",np.mean(pred==true))
    print(confusion_matrix(true,pred))
    print(classification_report(true,pred))