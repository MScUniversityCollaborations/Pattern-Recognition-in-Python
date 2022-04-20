import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Τάξη Linear Regression
class LinearRegression:

    # Αρχικοποίηση learning_rate και epochs (επαναλήψεις)
    # weights & bias = none
    # default επιλογές για learning_rate: 0,01 και epochs: 1000
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Training function: fit
    def fit(self, X, y):
        # shape of X:
        # αριθμός παραδειγμάτων εκπαίδευσης
        # αριθμός των χαρακτηριστικά
        number_of_training, number_of_features = X.shape

        # Αρχικοποίηση weights ως μήτρα μηδενικού μεγέθους
        # Βias = 0
        self.weights = np.zeros((number_of_features, 1))
        self.bias = 0

        # Αναδιαμόρφωση του y ως (number_of_training,1)
        # σε περίπτωση που το σύνολο δεδομένων μας αρχικοποιηθουν ως
        # (number_of_training,) όπου μπορεί να προκαλέσει προβλήματα
        y = y.values.reshape(number_of_training, 1)

        # Αρχικοποίηση άδειας λίστα losses για την αποθήκευση των σφαλμάτων
        losses = []

        # Αλγόριθμος Gradient Descent (Training loop)
        for epoch in range(self.epochs):
            # Υπολογισμός πρόβλεψης h(x) = y_hat
            y_hat = np.dot(X, self.weights) + self.bias

            # Υπολογισμός σφάλματος
            loss = np.mean((y_hat - y) ** 2)

            # Προσθήκη σφάλματος στη λίστα: losses
            losses.append(loss)

            # Υπολογισμός παραγώγων παραμέτρων (weights και bias)
            dw = (1 / number_of_training) * np.dot(X.T, (y_hat - y))
            db = (1 / number_of_training) * np.sum((y_hat - y))
            # Ενημέρωση των παραμέτρων
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # επιστροφή των παραμέτρων
        return self.weights, self.bias, losses


def input_data(bet):

    # Διαχωρισμός δεδομένων
    X, y, seed = bet.iloc[:, 1:], bet.iloc[:, 0], 40
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)

    # Κλήση της LinearRegression
    model = LinearRegression()
    w, b, losses = model.fit(X_train, y_train)

    # Επιστροφή μέσου σφάλματος με τη χρήση της numpy
    return np.mean(losses)

