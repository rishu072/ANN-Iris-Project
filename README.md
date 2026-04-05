# ANN-Iris-Project
A Deep Learning model built with TensorFlow/Keras to classify Iris species. Features data scaling, a 3-layer dense network, and accuracy evaluation.

# 🌸 ANN from Scratch using NumPy (Iris Dataset)

## 📌 Project Overview

This project implements an Artificial Neural Network (ANN) from scratch using only NumPy (no deep learning frameworks). The model is trained to classify Iris flower species into three categories:

* Setosa
* Versicolor
* Virginica

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn (only for comparison and evaluation)

---

## 📂 Project Structure

```
ANN/
│
├── data/
│   └── iris.data
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── main.py
├── README.md
├── report.pdf
```

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies

```
pip install numpy pandas matplotlib scikit-learn
```

### Step 2: Run the Code

```
python main.py
```

---

## 🧠 Model Architecture

* Input Layer: 4 neurons (features)
* Hidden Layer 1: 8 neurons (Sigmoid)
* Hidden Layer 2: 6 neurons (Sigmoid)
* Output Layer: 3 neurons (Softmax)

---

## 🔁 Training Details

* Loss Function: Cross-Entropy Loss
* Optimization: Gradient Descent
* Epochs: 500
* Learning Rate: 0.01

---

## 📊 Output

* Training Loss Curve 📉
* Accuracy (~90%+)
* Confusion Matrix
* Classification Report
* <img width="907" height="749" alt="image" src="https://github.com/user-attachments/assets/9032f9d9-855e-4eda-937a-87dfc4a3151c" />

---

## 🔬 Experimentation

Learning rates tested:

* 0.001
* 0.01
* 0.1

---

## ⚖️ Comparison

The model performance is compared with Scikit-learn’s MLPClassifier.

---

## 📌 Conclusion

The ANN model successfully classifies the Iris dataset with high accuracy. This project demonstrates a clear understanding of forward propagation, backpropagation, and neural network fundamentals.

---
