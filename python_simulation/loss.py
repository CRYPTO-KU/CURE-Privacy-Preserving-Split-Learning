import numpy as np
from decimal import getcontext

# loss function and its derivative
def mse(y_true, y_pred, precision):
    getcontext().prec = precision
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred, precision):
    getcontext().prec = precision
    return 2*(y_pred-y_true)/y_true.size
