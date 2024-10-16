import numpy as np
from decimal import getcontext

from loss import mse, mse_prime
from mat import add_random_noise
from activation import sigmoid, sigmoid_prime, approx_sigmoid, approx_sigmoid_prime

class Config:
    def __init__(self, input_num, output_num, epochs, learning_rate, hidden_layer_neurons, batch_size, cut, fl_point, FULLY_ENCRYPTED):
        self.input_num = input_num
        self.output_num = output_num
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_layer_neurons = hidden_layer_neurons 
        self.batch_size = batch_size
        self.cut = cut
        self.fl_point = fl_point
        self.FULLY_ENCRYPTED = FULLY_ENCRYPTED


ACTIVATION_MAP = {
    True: (approx_sigmoid, approx_sigmoid_prime),
    False: (sigmoid, sigmoid_prime)
}


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, fl_point, ENCRYPTED):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.fl_point = fl_point
        self.ENCRYPTED = ENCRYPTED

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = np.round(input_data, decimals=self.fl_point)
        self.output = np.dot(self.input, self.weights) + self.bias
        # Apply precision to the output
        self.output = np.round(self.output, decimals=self.fl_point)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        if self.ENCRYPTED == True:
            self.weights = add_random_noise(self.weights, 1e-6)
        self.bias -= learning_rate * output_error
        return input_error
    

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, fl_point, ENCRYPTED):
        self.activation, self.activation_prime = ACTIVATION_MAP[ENCRYPTED]
        self.fl_point = fl_point
        self.ENCRYPTED = ENCRYPTED

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = np.round(input_data, decimals=self.fl_point)
        self.output = self.activation(self.input, self.fl_point)
        self.output = np.round(self.output, decimals=self.fl_point)
        if self.ENCRYPTED == True:
            self.output = add_random_noise(self.output, 1e-6)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        getcontext().prec = 4
        return self.activation_prime(self.input, self.fl_point) * output_error
    

class Network:
    def __init__(self, layer_sizes, batch_size, learning_rate, fl_point, FULLY_ENCRYPTED):
        self.layers = []
        self.loss = None
        self.loss_prime = None

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        for i in range(len(layer_sizes) - 1):
            self.add_layer(FCLayer(layer_sizes[i], layer_sizes[i+1], fl_point, FULLY_ENCRYPTED))
            self.add_layer(ActivationLayer(fl_point, FULLY_ENCRYPTED))

    # add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, output):
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    # train the network
    def train_forward(self, output):
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
    
    def train_backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, self.learning_rate)
        return error
    

class SLNetwork:
    def __init__(self, config):
        self.loss = mse
        self.loss_prime = mse_prime
        self.config = config
        layer_sizes = [config.input_num] + config.hidden_layer_neurons + [config.output_num]

        server_layers = layer_sizes[:config.cut+1]
        client_layers = layer_sizes[config.cut:]

        self.server = Network(server_layers, config.batch_size, config.learning_rate, config.fl_point, True)
        self.client = Network(client_layers, config.batch_size, config.learning_rate, config.fl_point, config.FULLY_ENCRYPTED)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            output = self.server.predict(output)
            output = self.client.predict(output)
            result.append(output)

        return result

     # train the network
    def train(self, x_train, y_train, x_test, y_test):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(self.config.epochs):
            err = 0
            for j in range(0, samples, self.config.batch_size):
                # Extract a batch
                x_batch = x_train[j:j + self.config.batch_size]
                y_batch = y_train[j:j + self.config.batch_size]

                # Initialize batch error
                batch_err = 0

                # Batch training loop
                for k in range(len(x_batch)):
                    # forward propagation
                    output = x_batch[k]
                    output = self.server.train_forward(output)
                    output = self.client.train_forward(output)

                    # compute loss (for display purpose only)
                    batch_err += self.loss(y_batch[k], output, self.config.fl_point)

                    # backward propagation
                    error = self.loss_prime(y_batch[k], output, self.config.fl_point)
                    error = self.client.train_backward(error)
                    error = self.server.train_backward(error)

                # calculate average error on the batch
                batch_err /= len(x_batch)
                err += batch_err

            # calculate average error on all samples
            err /= (samples // self.config.batch_size)
            print('epoch %d/%d   error=%f' % (i + 1, self.config.epochs, err))
            epoch_accuracy = self.test(x_test, y_test)
            print(f"Epoch Accuracy: {epoch_accuracy * 100:.2f}%")

    def test(self, x, y):
        correct = 0
        total = len(x)

        for i in range(total):
            output = self.predict(x[i])[0]
            predicted_label = np.argmax(output)
            true_label = np.argmax(y[i])

            if predicted_label == true_label:
                correct += 1

        return correct / total
