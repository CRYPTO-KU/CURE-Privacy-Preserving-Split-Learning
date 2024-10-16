import numpy as np
from decimal import getcontext

interval = 15
degree = 7

# Sigmoid activation function and its derivative
def sigmoid(x, precision):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x, precision):
    return sigmoid(x, precision) * (1 - sigmoid(x, precision))

def approx_sigmoid(x, precision):
    getcontext().prec = precision
    if degree == 5:
        if interval == 10:
            return 0.5 + 0.177267 * x + 5.02622e-17 * np.power(x,2) - 0.00312445 * np.power(x,3) - 6.43682e-19 * np.power(x,4) + 1.91422e-05 * np.power(x,5)
        if interval == 15:
            return 0.5 + 0.141532 * x + 1.10885e-17 * np.power(x,2) - 0.00129443 * np.power(x,3) - 6.04788e-20 * np.power(x,4) + 3.76227e-06 * np.power(x,5)
    
    if degree == 7:
        if interval == 10:
            return 0.5 + 0.204371 * x - 1.37033e-16 * np.power(x,2) - 0.00596296 * np.power(x,3) + 4.111e-18 * np.power(x,4) + 8.39323e-05 * np.power(x,5) - 3.04518e-20 * np.power(x,6) - 4.00743e-07 * np.power(x,7)
        if interval == 15:
            return 0.5 + 0.169698 * x + 5.58284e-17 * np.power(x,2) - 0.00272638 * np.power(x,3) - 7.51897e-19 * np.power(x,4) + 1.87701e-05 * np.power(x,5) + 2.49518e-21 * np.power(x,6) - 4.19623e-08 * np.power(x,7)

    if degree == 9:
        if interval == 10:
            return 0.5 + 0.221879 * x + 1.82216e-17 * np.power(x,2) - 0.00896839 * np.power(x,3) + 2.63291e-19 * np.power(x,4) + 0.000208218 * np.power(x,5) - 1.77919e-20 * np.power(x,6) - 2.21104e-06 * np.power(x,7) + 1.44224e-22 * np.power(x,8) + 8.55614e-09 * np.power(x,9)
        if interval == 15:
            return 0.5 + 0.191025 * x + 1.20976e-16 * np.power(x,2) - 0.00451544 * np.power(x,3) - 3.02706e-18 * np.power(x,4) + 5.29761e-05 * np.power(x,5) + 2.3813e-20 * np.power(x,6) - 2.68244e-07 * np.power(x,7) - 5.81546e-23 * np.power(x,8) + 4.81871e-10 * np.power(x,9)

def approx_sigmoid_prime(x, precision):
    getcontext().prec = precision
    if degree == 5:
        if interval == 10:
            return 0.183357 - 7.4476e-18 * x - 0.00863088 * np.power(x, 2) + 7.4476e-20 * np.power(x, 3) + 7.46419e-05 * np.power(x, 4)
        if interval == 15:
            return 0.162938 + 1.38262e-18 * x - 0.00407653 * np.power(x, 2) - 8.61215e-21 * np.power(x, 3) + 1.66191e-05 * np.power(x, 4)

    if degree == 7:
        if interval == 10:
            return 0.205886 + 4.96643e-18 * x - 0.0168043 * np.power(x, 2) - 1.06344e-19 * np.power(x, 3) + 0.00035135 * np.power(x, 4) + 6.06453e-22 * np.power(x, 5) - 2.08302e-06 * np.power(x, 6)
        if interval == 15:
            return 0.182093 - 9.25527e-18 * x - 0.00842343 * np.power(x, 2) + 5.85435e-20 * np.power(x, 3) + 8.60489e-05 * np.power(x, 4) - 7.60677e-23 * np.power(x, 5) - 2.37999e-07 * np.power(x, 6)
    
    if degree == 9:
        if interval == 10:
            return 0.221713 - 6.15976e-18 * x - 0.0255962 * np.power(x,2) - 2.66069e-19 * np.power(x, 3) + 0.000908776 * np.power(x, 4) + 9.7006e-21 * np.power(x, 5) - 1.22751e-05 * np.power(x, 6) - 6.53006e-23 * np.power(x, 7) + 5.5536e-08 * np.power(x, 8)
        if interval == 15:
            return 0.198112 - 2.65246e-19 * x - 0.0137036 * np.power(x,2) + 6.11457e-20 * np.power(x,3) + 0.000246214 * np.power(x,4) - 6.8972e-22 * np.power(x,5) - 1.58287e-06 * np.power(x,6) + 1.83418e-24 * np.power(x,7) + 3.3175e-09 * np.power(x,8)
