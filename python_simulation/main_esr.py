import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from net import SLNetwork, Config

def parse_hidden_layers(hidden_layers_str):
    hidden_layers = hidden_layers_str.split(",")
    neurons = [int(s) for s in hidden_layers]
    return neurons

# Load dataset
data = pd.read_csv("../data/Epileptic Seizure Recognition.csv")

# Data Frame
df = pd.DataFrame(data)

# drop id , Unnamed: 32 columns
columns_to_drop = ["Unnamed"]

for column in columns_to_drop:
    df.drop(column, axis=1, inplace=True)

X = df.drop(columns='y')
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.array(df["y"])
y -= 1

X_train, X_test = X[:9200,:], X[9200:,:]
y_train, y_test = y[:9200], y[9200:]

X_train = X_train.reshape(X_train.shape[0], 1, 178)
y_train = np.eye(5)[y_train.astype(int)]

X_test = X_test.reshape(X_test.shape[0], 1, 178)
y_test = np.eye(5)[y_test.astype(int)]

random.seed(time.time())

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=int, default=178, help="input controls the number of input nodes")
parser.add_argument("--hidden", type=str, default="128,32", help="output controls the number of hidden nodes (comma-separated)")
parser.add_argument("--output", type=int, default=5, help="output controls the number of output nodes")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
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
sl_network.train(X_train, y_train, X_test, y_test)

# After training, you can use the accuracy method to evaluate the network on a dataset.
test_accuracy = sl_network.test(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
