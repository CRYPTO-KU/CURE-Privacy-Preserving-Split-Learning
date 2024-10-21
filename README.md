# CURE: Privacy-Preserving Split Learning

#### Forked from https://github.com/PaluMacil/gophernet.


## Overview

This repository contains the implementation of the CURE system, a novel approach to privacy-preserving split learning. CURE leverages homomorphic encryption (HE) to secure the server side of the model and optionally the data, enabling secure split learning without compromising training efficiency.

## Features

- **Homomorphic Encryption**: Utilizes HE to encrypt server-side model parameters, enhancing data privacy while allowing computation on encrypted data.
- **Efficient Computation**: Implements advanced packing techniques to optimize communication and computational overhead, making it feasible for practical applications.
- **Flexibility**: Supports various configurations for different layers and privacy needs, adaptable to both simple and complex neural network architectures.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/CURE-repo.git
   ```
2. **Setup Environment**:
   Ensure you have Go installed on your machine, as the main implementation is in Go. Python is required for some utility scripts.

### Data Preparation

Data is prepared and loaded through a simple command-line utility that formats the raw datasets for use in training. Ensure your datasets are in the required format and located in the `data/` directory within the project structure.

### Download data

Train set: https://pjreddie.com/media/files/mnist_train.csv

Test set: https://pjreddie.com/media/files/mnist_test.csv

Place as `data/mnist_train.csv` and `data/mnist_test.csv`.


3. **Run Experiments**:
   Navigate to the project directory and execute:
   ```bash
   go run main.go train digits -layers=4 -hidden=s1,s2,_,sn,c,c1,c2,_,ck -epochs=10 -rate=.1
   ```
   Replace `s1,s2,_,sn` with your server-side configuration parameters, `c1,c2,_,ck` with your client-side configuration parameters, use flags `"data amount"` with the amount of data to be processed, and `"batch_size"` with the size of the batch for processing.

## Usage

- **Training**: To train the model using the predefined configurations, use the command-line arguments to specify parameters such as the number of epochs, learning rate, and batch size.
- **Prediction**: Run the prediction module with the trained model to evaluate its performance on new data.

## Python Simulations

The `python_simulation` code trains simple neural networks on the MNIST, BCW (Breast Cancer Wisconsin), and CREDIT datasets using a configurable architecture. This section allows for quick experimentation with different configurations and fast accuracy assessments.

To run any of the main scripts, use:
```bash
python main.py --input 784 --hidden "128,32" --output 10 --epochs 10 --rate 0.01 --batch 60
```
## Time Latency Simulations

The tests_for_time folder contains the code for time simulations conducted in Go for a specific split learning setup. This section enables users to determine the amount of time required for a given set of network configuration parameters.

For example, an MNIST network case can be run with the following command, using 10,000 data points, a batch size of 60, and an activation function degree of 3 with first hidden layer belonging to server:
```bash
go run main.go 784,128,c,32,10 10000 60 3
```

## Advisor for Optimal Layer Partitioning

The **Advisor** function helps determine the optimal split point between the client and server in split learning by using estimations based on approximations. These estimations balance computational load, memory usage, and communication overhead, providing an efficient partitioning strategy that optimizes performance while considering both server and client system constraints.

### Usage

To use the Advisor function, simply run the following command:

```bash
go run main.go -cpu=40
```

## Contribution

Contributions are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation
Please cite this work as:
```
@article{kanpak2024cure,
    author       = {Halil Ibrahim Kanpak and Aqsa Shabbir and Esra Genç and Alptekin Küpçü and Sinem Sav},
    title        = {{CURE: Privacy-Preserving Split Learning Done Right}},
    journal      = {arXiv preprint arXiv:2407.08977},
    year         = {2024},
    url          = {https://doi.org/10.48550/arXiv.2407.08977}
}
```

## Contact

For any questions or concerns, please open an issue in this repository.
