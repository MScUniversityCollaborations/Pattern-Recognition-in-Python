import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Inputs / Outputs / Weights
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])
weights = np.array([0.0, 0.0])
learning_rate = 0.1

scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)


# Step Function
def step_function(sum):
    if sum >= 1:
        return 1
    return 0


# Calculate Output
def calculate_output(instance):
    s = instance.dot(weights)
    return step_function(s)


def train():
    total_error = 1
    while total_error != 0:
        total_error = 0
        for i in range(len(outputs)):
            prediction = calculate_output(inputs[i])
            error = abs(outputs[i] - prediction)
            total_error += error
            if error > 0 :
                for j in range(len(weights)):
                    weights[j] = weights[j] + (learning_rate * inputs[i][j] * error)
                    print('Weight update: ' + str(weights[j]))
        print('Total error: ' + str(total_error))


def start():
    print(train())
