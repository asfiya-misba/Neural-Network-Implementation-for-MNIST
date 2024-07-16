# Assignment 2
# Summer 2023
# Asfiya Misba - 1002028239
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from NeuralNetwork_Lib import *

# Loading the digits dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocessing the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
input_size = X_train.shape[1]
output_size = y_train.shape[1]

# Defining parameters for different models
hyperparameters = [
    {'num_layers': 10, 'nodes_per_layer': 100, 'activation': 'Sigmoid',
     'learning_rate': 0.01, 'batch_size': 128, 'max_epochs': 10, 'early_stopping': 5},
    {'num_layers': 100, 'nodes_per_layer': 60, 'activation': 'HyperbolicTangent',
     'learning_rate': 0.1, 'batch_size': 128, 'max_epochs': 10, 'early_stopping': 5},
    {'num_layers': 10, 'nodes_per_layer': 60, 'activation': 'Sigmoid',
     'learning_rate': 0.5, 'batch_size': 128, 'max_epochs': 10, 'early_stopping': 5},
]

# Training the model based on the parameters
for i, params in enumerate(hyperparameters):

    # Creating the model
    model = Sequential()

    # Adding layers to the model
    model.add(LinearLayer(X_train.shape[1], params['nodes_per_layer']))
    if params['activation'] == 'Sigmoid':
        model.add(SigmoidLayer())
    elif params['activation'] == 'HyperbolicTangent':
        model.add(HyperbolicTangentLayer())
    else:
        raise ValueError("Invalid activation function specified.")

    for _ in range(params['num_layers'] - 2):
        model.add(LinearLayer(params['nodes_per_layer'], params['nodes_per_layer']))
        if params['activation'] == 'Sigmoid':
            model.add(SigmoidLayer())
        elif params['activation'] == 'HyperbolicTangent':
            model.add(HyperbolicTangentLayer())
        else:
            raise ValueError("Invalid activation function specified.")

    model.add(LinearLayer(params['nodes_per_layer'], 10))
    model.add(SoftmaxLayer())

    # Defining the loss function
    model.loss = CrossEntropyLoss()

    print(f"Model {i + 1}:")
    print(f"Activation Function: {params['activation']}")
    print(f"Number of Layers: {params['num_layers']}")
    print(f"Nodes per Layer: {params['nodes_per_layer']}")
    print(f"Learning Rate: {params['learning_rate']}")
    print(f"Batch Size: {params['batch_size']}")
    print(f"Max Epochs: {params['max_epochs']}")
    print(f"Early Stopping: {params['early_stopping']}")
    train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val, params['learning_rate'],
                                         params['max_epochs'], params['batch_size'], params['early_stopping'])

    # Plotting the graph
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Model {i + 1} Training and Validation Loss")
    plt.legend()
    plt.show()

    # Accuracy of the model
    test_accuracy = model.Predict(X_test, y_test)
    print(f"Model {i + 1} Test Accuracy: {test_accuracy * 100:.2f}")
    print('------------------------------------')

    # Saving the weights
    model.save_weights(f'MNIST_model{i + 1}.w')
