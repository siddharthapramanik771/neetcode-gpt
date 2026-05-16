import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))

        # Loss: L = 0.5 * (y_hat - y_true)^2
        L = 0.5 * (y_hat - y_true)**2
        dL_dy = y_hat - y_true
        # y_hat = 1/(1+e^-z)
        # dy_dz = -1((1+e^-z)^-2) * (-e^-z) = (e^-z) / (1+e^-z)^2
        # dy_dz = {1 - 1/(1+e^-z)} * {1/(1+e^-z)}
        # dy_dz = (1-y)*y
        dy_dz = y_hat*(1.0 - y_hat)
        dz_dw = x
        dz_db = 1

        dL_dw = dL_dy * dy_dz * dz_dw
        dL_db = dL_dy * dy_dz * dz_db
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        return (np.round(dL_dw, 5), round(float(dL_db), 5))
