import numpy as np

inputs = [20, 23]
weights = [-0.9, 0.1]


def sum_function(inputs, weights):
    s = 0
    for i in range(2):
        s += inputs[i] * weights[i]
    return s


s = sum_function(inputs, weights)


def step_function(sum):
    if sum >= 1:
        return 1
    return 0


def start():
    print(step_function(s))
