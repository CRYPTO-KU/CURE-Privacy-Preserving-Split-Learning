import os
import time
import random
import argparse
import numpy as np

from net import SLNetwork, Config

def parse_hidden_layers(hidden_layers_str):
    hidden_layers = hidden_layers_str.split(",")
    neurons = [int(s) for s in hidden_layers]
    return neurons

# Load MNIST data from CSV files
train_data = np.loadtxt("../data/mnist_train.csv", delimiter=",", skiprows=0)
test_data = np.loadtxt("../data/mnist_test.csv", delimiter=",", skiprows=1)

x_train = train_data[:, 1:]  # Exclude the first column (labels)
y_train = train_data[:, 0]   # Use the first column as labels

x_test = test_data[:, 1:]    # Exclude the first column (labels)
y_test = test_data[:, 0]     # Use the first column as labels

# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float64')
x_train /= 255
x_train *= 0.99
x_train += 0.01

# One-hot encode the labels
y_train = np.eye(10)[y_train.astype(int)]

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float64')
x_test /= 255
x_test *= 0.99
x_test += 0.01
y_test = np.eye(10)[y_test.astype(int)]

random.seed(time.time())

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=int, default=784, help="input controls the number of input nodes")
parser.add_argument("--hidden", type=str, default="128,32", help="output controls the number of hidden nodes (comma-separated)")
parser.add_argument("--output", type=int, default=10, help="output controls the number of output nodes")
parser.add_argument("--epochs", type=int, default=6, help="number of epochs")
parser.add_argument("--rate", type=float, default=0.01, help="rate is the learning rate")
parser.add_argument("--batch", type=int, default=60, help="batch size")
parser.add_argument("--cut", type=int, default=2, help="cut layer")
parser.add_argument("--flpoint", type=int, default=4, help="fixed presicion")
parser.add_argument("--fully_encrypted", type=bool, default=False, help="if false, only server side is encrypted")

args = parser.parse_args(os.sys.argv[1:])

hidden_layer_neurons = parse_hidden_layers(args.hidden)

config = Config(
    input_num = args.input,
    output_num = args.output,
    epochs = args.epochs,
    learning_rate = args.rate,
    hidden_layer_neurons = hidden_layer_neurons,
    batch_size = args.batch,
    cut = args.cut,
    fl_point = args.flpoint,
    FULLY_ENCRYPTED = args.fully_encrypted
)

sl_network = SLNetwork(config)

# train on samples
sl_network.train(x_train, y_train, x_test, y_test)

# After training, you can use the accuracy method to evaluate the network on a dataset.
test_accuracy = sl_network.test(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
