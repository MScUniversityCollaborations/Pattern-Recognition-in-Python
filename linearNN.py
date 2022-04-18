# Simple Perceptron
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
#
# # Inputs / Outputs / Weights
# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# outputs = np.array([0, 0, 0, 1])
# weights = np.array([0.0, 0.0])
# learning_rate = 0.1
# scaler = MinMaxScaler()
# inputs = scaler.fit_transform(inputs)
#
#
# # Step Function
# def step_function(sum):
#     if sum >= 1:
#         return 1
#     return 0
#
#
# # Calculate Output
# def calculate_output(instance):
#     s = instance.dot(weights)
#     return step_function(s)
#
#
# def train(inputs):
#     total_error = 1
#     while total_error != 0:
#         total_error = 0
#         for i in range(len(outputs)):
#             prediction = calculate_output(inputs[i])
#             error = abs(outputs[i] - prediction)
#             total_error += error
#             if error > 0:
#                 for j in range(len(weights)):
#                     weights[j] = weights[j] + (learning_rate * inputs[i][j] * error)
#                     print('Weight update: ' + str(weights[j]))
#         print('Total error: ' + str(total_error))

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from statsmodels.tools.eval_measures import rmse
import numpy as np
import matplotlib.pyplot as plt

# Linear Regression class
class LinReg:

    # Initializing lr: learning rate, epochs: no. of iterations,
    # weights & bias: parameters as None    # default lr: 0.01, epochs: 800
    def __init__(self, lr=0.01, epochs=800):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None  # Training function: fit


    def fit(self, X, y):
        # shape of X: (number of training examples: m, number of
        # features: n)
        m, n = X.shape

        # Initializing weights as a matrix of zeros of size: (number
        # of features: n, 1) and bias as 0
        self.weights = np.zeros((n, 1))
        self.bias = 0

        # reshaping y as (m,1) in case your dataset initialized as
        # (m,) which can cause problems
        y = y.values.reshape(m, 1)

        # empty lsit to store losses so we can plot them later
        # against epochs
        losses = []

        # Gradient Descent loop/ Training loop
        for epoch in range(self.epochs):
            # Calculating prediction: y_hat or h(x)
            y_hat = np.dot(X, self.weights) + self.bias

            # Calculting loss
            loss = np.mean((y_hat - y) ** 2)

            # Appending loss in list: losses
            losses.append(loss)

            # Calculating derivatives of parameters(weights, and
            # bias)
            dw = (1 / m) * np.dot(X.T, (y_hat - y))
            db = (1 / m) * np.sum((y_hat - y))  # Updating the parameters: parameter := parameter - lr*derivative
            # of loss/cost w.r.t parameter)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        # returning the parameter so we can look at them later
        return self.weights, self.bias, losses  # Predicting(calculating y_hat with our updated weights) for the


    # testing/validation
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def start(bet):

    X, y, seed = bet.iloc[:, 1:], bet.iloc[:, 0], 40

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    model = LinReg(epochs=100)
    w, b, l = model.fit(X_train, y_train)
    print("W: ", w)
    print("b: ", b)
    print("l: ", l)
    print("mean l: ", np.mean(l))

    # Plotting our predictions.
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(X, y)
    # plt.plot(X, model.predict(X))  # X and predictions.
    # plt.title('Hours vs Percentage')
    # plt.xlabel('X (Input) : Hours')
    # plt.ylabel('y (Target) : Scores')
