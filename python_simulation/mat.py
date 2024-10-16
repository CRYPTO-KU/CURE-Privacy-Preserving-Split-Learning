import numpy as np
import time

def add_random_noise(matrix, noise_amplitude):
    np.random.seed(int(time.time()))

    noise = np.random.rand(*matrix.shape) * 2 * noise_amplitude - noise_amplitude
    noisy_matrix = matrix + noise

    return noisy_matrix
