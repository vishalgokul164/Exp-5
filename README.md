# EXPERIMENT 5 : MULTILAYER PERCEPTRON (MLP) NEURAL NETWORK

## AIM

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> *Note:* The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP
> experiment.

## OVERVIEW

A *Multilayer Perceptron (MLP)* is a feed-forward artificial neural
network consisting of an input layer, one or more hidden layers, and an
output layer. It learns patterns in data by adjusting weights and biases
during training using backpropagation.

In this experiment, the MLP model is trained on the *Digits dataset*
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## ALGORITHM

1.  *Start.*
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. *Stop.*
ry slightly with changes to the train-test split,
    random seed, or model parameters.


## PROGRAM

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.datasets import load_digits  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier  
from sklearn.preprocessing import StandardScaler  
print("--- Dataset Initialization ---")  
digits = load_digits()  
X = digits.data  
y = digits.target  
print(f"Dataset shape: {X.shape}")  
print(f"Number of classes: {len(np.unique(y))} (Digits 0-9)")    
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42  
)  
print(f"Training set size: {X_train.shape[0]} samples")  
print(f"Test set size: {X_test.shape[0]} samples")  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)    
X_test_scaled = scaler.transform(X_test)  
print("\n--- Model Training ---")  
mlp = MLPClassifier(  
    hidden_layer_sizes=(64,),    
    activation="relu",  
    solver="adam",  
    max_iter=500,  
    random_state=42,  
)  
mlp.fit(X_train_scaled, y_train)  
print("MLP Classifier trained successfully.")    
print(f"Number of iterations until convergence: {mlp.n_iter_}")  
y_pred = mlp.predict(X_test_scaled)  
print("\n--- Evaluation Metrics ---")  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Test Accuracy: {accuracy:.4f}")  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  
print("\n--- Generating Training Loss Curve ---")  
plt.figure(figsize=(8, 5))  
plt.plot(mlp.loss_curve_, marker="o", color="teal", linewidth=2)  
plt.title("MLP Training Loss Curve")  
plt.xlabel("Iterations / Epochs")  
plt.ylabel("Loss")  
plt.grid(True, linestyle="--", alpha=0.6)  
plt.tight_layout()  
print("Training loss curve plotted successfully.")  
plt.show()        


## OUTPUT

<img width="440" height="513" alt="image" src="https://github.com/user-attachments/assets/510623e1-3bf2-4271-abd3-fcaa38b6ec01" />

<img width="735" height="457" alt="image" src="https://github.com/user-attachments/assets/ccc1937a-66df-4a46-b64e-25c893ddf054" />

## RESULT

Thus to implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset 
