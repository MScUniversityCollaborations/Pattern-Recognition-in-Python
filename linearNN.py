import numpy as np

inputs = np.array([20, 23])
weights = np.array([0.9, 0.1])


def sum_function(inputs, weights):
    return inputs.dot(weights)


s = sum_function(inputs, weights)


def step_function(sum):
    if sum >= 1:
        return 1
    return 0


def start():
    print(step_function(s))
