# Import required modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold


# Defining the class
class LinearRegression:

    def __init__(self, x, y):
        self.data = x
        self.label = y
        self.m = 0
        self.b = 0
        self.n = len(x)

    def fit(self, epochs, lr):
        # Implementing Gradient Descent
        for i in range(epochs):
            y_pred = self.m * self.data + self.b

            # Calculating derivatives w.r.t Parameters
            D_m = (-2 / self.n) * sum(self.data * (self.label - y_pred))
            D_b = (-1 / self.n) * sum(self.label - y_pred)

            # Updating Parameters
            self.m = self.m - lr * D_m
            self.c = self.b - lr * D_b

    def predict(self, inp):
        y_pred = self.m * inp + self.b
        return y_pred


def start(bet):

    # Preparing the data
    x = np.array(bet.iloc[:, 0])
    y = np.array(bet.iloc[:, 1])

    # Creating the class object
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    regressor = LinearRegression(x, y)

    # Training the model with .fit method
    regressor.fit(1000, 0.0001)  # epochs-1000 , learning_rate - 0.0001

    # 10 Fold validation and creating error array
    cv, error = KFold(n_splits=10), []
    for train_idx, valid_idx in cv.split(x):
        # Splitting data
        split_X_train, split_X_test = x[train_idx], x[valid_idx]
        split_Y_train, split_Y_test = y[train_idx], y[valid_idx]



    for i in range(0, 2):
        if (i == 0):
            regressor = LinearRegression(x_train, y_train)
            regressor.fit(1000, 0.0001)
            y_pred = regressor.predict(x)
            # print(y_pred)
        if (i == 1):
            regressor = LinearRegression(x_test, y_test)
            regressor.fit(1000, 0.0001)
            y_pred = regressor.predict(x)
            # print(y_pred)


    # # Prediciting the values
    # y_pred = regressor.predict(x)
    # # print(y_pred)
    #
    # #Plotting the results
    # plt.figure(figsize = (10,6))
    # plt.scatter(x, y, color = 'green')
    # plt.plot(x, y_pred, color='k', lw=3)
    # plt.xlabel('x', size=20)
    # plt.ylabel('y', size=20)
    # plt.show()