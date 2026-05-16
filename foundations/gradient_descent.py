class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        def f(x):
            return x**2
        # Derivative:         f'(x) = 2x
        def f_1(x):
            return 2*x
        # Update rule:        x = x - learning_rate * f'(x)
        x =  init
        for i in range(iterations):
            x = x - learning_rate * f_1(x)
        # Round final answer to 5 decimal places
        return round(x,5)
