import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Preprocess the text data using CountVectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(X)

        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training the Perceptron
        for _ in range(self.n_iters):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.weights) + self.bias) <= 0:
                    self.weights += self.lr * y[i] * X[i]
                    self.bias += self.lr * y[i]

    def predict(self, X):
        # Preprocess input text
        X = vectorizer.transform(X)

        # Make predictions
        predictions = np.sign(np.dot(X, self.weights) + self.bias)
        return predictions


# Sample training data
X_train = [
    "I loved this movie, it was so much fun!",
    "The food at this restaurant is not good. Don't go there!",
    "The new iPhone looks amazing, can't wait to get my hands on it."
]
y_train = [1, -1, 1]

# Initialize and train the Perceptron model
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Sample test data
X_test = [
    "This is a great product, I highly recommend it.",
    "I had a terrible experience with their customer service."
]

# Make predictions
predictions = perceptron.predict(X_test)
print("Predictions:", predictions)
