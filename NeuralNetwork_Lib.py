# Assignment 2
# Summer 2023
# Asfiya Misba - 1002028239
import numpy as np
import pickle

from sklearn.metrics import accuracy_score


# Layer class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_error, learning_rate):
        pass


# Linear Layer class
class LinearLayer(Layer):
    def __init__(self, input_size, output_size, weights=0.5):
        super().__init__()

        if weights is None:
            self.weights = np.zeros((input_size, output_size))
            self.bias = np.zeros((1, output_size))

        elif weights == 10:
            self.weights = np.random.uniform(-10, 10, size=(input_size, output_size))
            self.bias = np.random.uniform(1, output_size)

        elif weights == 0.5:
            self.weights = np.random.rand(input_size, output_size) - 0.5
            self.bias = np.random.rand(1, output_size) - 0.5

        # self.weights = np.random.randn(input_size, output_size)
        # self.bias = np.random.randn(output_size)

    def forward(self, input, target=None):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error.sum(axis=0)
        return input_error


# Sigmoid Layer class
class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error * self.output * (1 - self.output)


# Hyperbolic Tangent Layer class
class HyperbolicTangentLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.tanh(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error * (1 - np.square(self.output))


# Softmax Layer class
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error, learning_rate):
        gradient = 1 / (1 + np.exp(-self.input))
        return gradient * (1 - gradient) * output_error


# Cross-Entropy Loss class
class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        self.input = input
        self.target = target
        with np.errstate(over='ignore'):
            self.output = -np.sum(target * np.log(input)) / input.shape[0]
        return self.output

    def backward(self, learning_rate):
        return self.input - self.target


# Sequential class
class Sequential(Layer):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        self.output = input
        return self.output

    def Predict(self, X_test, y_test):
        output = self.forward(X_test)
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = 1 - accuracy_score(true_labels, predictions)
        return accuracy

    '''
    def backward(self, learning_rate):
        output_error = 1
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)
        return output_error
    '''

    def backward(self, output_error, learning_rate):
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def fit(self, X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size, early_stopping):
        train_losses = []  # To store training loss
        val_losses = []  # To store validation loss
        stop_counter = 0  # Counter used to track early stopping
        best_loss_value = float('inf')
        for epoch in range(epochs):
            for batch in range(0, len(X_train), batch_size):
                stop_batch = batch + batch_size
                batch_X = X_train[batch:stop_batch]
                batch_y = y_train[batch:stop_batch]

                # Forward pass
                output = self.forward(batch_X)
                loss = self.loss.forward(output, batch_y)
                # accuracy = accuracy_score(np.argmax(batch_y, axis=1), np.argmax(output, axis=1))

                # Backward pass
                output_error = self.loss.backward(learning_rate)
                self.backward(output_error, learning_rate)

            # Computing loss
            val_output = self.forward(X_val)
            val_loss = self.loss.forward(val_output, y_val)

            # To check for early stopping
            if val_loss < best_loss_value:
                best_loss_value = val_loss
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= early_stopping:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break

            '''
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f} - "
                  f"Val Loss: {val_loss:.4f}")
            '''

            train_losses.append(loss)
            val_losses.append(val_loss)
        return train_losses, val_losses

    def accuracy_score(self, y_pred, y_output):
        return np.mean(y_pred == y_output)

    def Predict1(self, ip_data, prob=False):
        samples = len(ip_data)
        result = []
        y_pred = []
        for i in range(samples):
            output = ip_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        if prob:
            return result

        for val in result:
            if val[0][0] >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return np.array(y_pred)

    # To save weights
    def save_weights(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.layers, file)

    # To load weights
    def load_weights(self, filename):
        with open(filename, 'rb') as file:
            self.layers = pickle.load(file)
