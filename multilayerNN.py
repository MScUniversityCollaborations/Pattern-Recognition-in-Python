import numpy as np
import nnfs

nnfs.init()


# Τάξη Dense layer
class Layer_Dense:

    # Αρχικοποίηση επιπέδου (layer)
    def __init__(self, n_inputs, n_neurons):
        # Αρχικοποίηση weights and biases
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Υπολογισμός των τιμών της εξόδου από τις εισόδους (weights και biases)
        self.output = np.dot(inputs, self.weights) + self.biases


# Τάξη ReLU activation
class Activation_ReLU:

    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # Υπολογισμός των τιμών της εξόδου από τις εισόδους
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # Παίρνουμε τις μη κανονικές πιθανότητες
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Τις κανονικοποιούμε για κάθε δείγμα
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities


# Τάξη Loss (Common loss class)
class Loss:

    # Υπολογίζουμε τις απώλειες από τα δεδομένα μας
    # και τα τακτοποιούμε βάση της εξόδου του μοντέλου μας
    def calculate(self, output, y):
        # Υπολογισμός των απωλειών δείγματος
        sample_losses = self.forward(output, y)

        # Υπολογισμός μέσης απώλειας
        data_loss = np.mean(sample_losses)

        # Επιστρέφουμε το σφάλμα
        return data_loss


# Τάξη Loss Categorical Cross Entropy (Cross-entropy loss)
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Αριθμός δειγμάτων σε μία παρτίδα
        samples = len(y_pred)

        # Αποκοπή δεδομένων για την αποτροπή της διαίρεσης τους με το μηδέν
        # Επίσης γίνεται αποκοπή κι από τις δύο πλευρές για να μην επηρεαστεί κάποια τιμή
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Έλεγχος έαν πετύχαμε τον στόχο μας εφόσον υπάρχουν categorical ετικέτες
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Τιμές μάσκας - μόνο για κωδικοποιημένες ετικέτες που έχουν ένα μόνο χαρακτηριστικό
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Απώλειες (Losses)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


def input_data(bet, input_layer):
    # Δημιουργία δεδομένων (dataset)
    train = len(bet)  # Δεδομένα εκπαίδευσης

    X, y = bet.iloc[:train, 1:], bet.iloc[:train, 0]

    # Δημιουργία μοντέλου
    dense1 = Layer_Dense(input_layer, 10)  # πρώτο dense layer
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(10, 3)  # δεύτερο dense layer
    activation2 = Activation_Softmax()

    # Δημιουργία loss συνάρτησης
    loss_function = Loss_CategoricalCrossentropy()

    # Βοηθητικές μεταβλητές
    lowest_loss = 9999999  # δίνουμε μία αρχική τιμή
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    for iteration in range(1000):

        # ενημέρωση weights με τυχαίες μικρές τιμές
        dense1.weights += 0.05 * np.random.randn(input_layer, 1)
        dense1.biases += 0.05 * np.random.randn(1, 10)
        dense2.weights += 0.05 * np.random.randn(10, 1)
        dense2.biases += 0.05 * np.random.randn(1, 1)

        # Κάνουμε ένα πέρασμα (forward pass) των δεδομένων μας από τα επίπεδα
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Παίρνουμε την απώλεια που επιστρέφει το δεύτερο στρώμα
        loss = loss_function.calculate(activation2.output, y)

        # Υπολογίζουμε τις τιμές του πρώτου axis
        # Υπολογίζουμε το accuracy από τη δεύτερη έξοδο ενεργοποίησης
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        # Έλεγχος εάν η απώλεια είναι η μικρότερη
        # και αποθηκεύουμε τα weights και τα bias
        if loss < lowest_loss:
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            final_accuracy = accuracy
        # Επαναφορά των weights και των biases
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()

    # Επιστρέφουμε το τελικό accuracy και τό lowest_loss
    return final_accuracy, lowest_loss
