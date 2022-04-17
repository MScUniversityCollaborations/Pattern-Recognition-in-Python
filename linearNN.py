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



def start(bet):

    X, y = bet.iloc[:, 1:], bet.iloc[:, 0]

    scores = []  # to store r squared
    rmse_list = []  # to store RMSE
    lrmodel = LinearRegression()
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        # X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

        lrmodel.fit(X_train, y_train)
        y_predict = lrmodel.predict(X_test)
        scores.append(lrmodel.score(X_test, y_test))
        rmse_fold = rmse(y_test, y_predict)
        rmse_list.append(rmse_fold)

    X1 = bet.iloc[:, :-2].values
    y1 = bet.iloc[:, -1].values
    lrmodel1 = LinearRegression()
    for train_index, test_index in cv.split(X1):
        # X_train1, X_test1, y_train1, y_test1 = X1[train_index], X1[test_index], y1[train_index], y1[test_index]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.15)
        fit = lrmodel1.fit(X_train1, y_train1)

    # print(fit)