
# Class_Ex1:
# Lets consider the 2 following sentences
# Sentence 1: I  am excited about the perceptron network.
# Sentence 2: we will not test the classifier with real data.
# Design your bag of words set and create your input set.
# Choose your BOW words that suits perceptron network.
# Design your classes that Sent 1 has positive sentiment and sent 2 has a negative sentiment.

# ----------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Given sentences
sentences = [
    "I am excited about the perceptron network.",
    "We will not test the classifier with real data."
]

# Labels for sentiment
labels = ["positive", "negative"]

# Create a Bag of Words (BOW) set
bow_set = ["excited", "perceptron", "network", "test", "classifier", "real", "data", "not"]

# Initialize CountVectorizer with the BOW set
vectorizer = CountVectorizer(vocabulary=bow_set)

# Vectorize the sentences
X = vectorizer.transform(sentences)

y = labels

for i in range(len(sentences)):
    print(f"Sentence: '{sentences[i]}'")
    print(f"BOW representation: {X[i].toarray()}")
    print(f"Sentiment: {y[i]}\n")


---------------------------------------------------------------
# Class_Ex2_1:

# For preprocessing, the text data is vectorized into feature vectors using a bag-of-words approach.
# Each sentence is converted into a vector where each element represents the frequency of a word from the vocabulary.
# This allows the textual data to be fed into the perceptron model.

# The training data consists of sample text sentences and corresponding sentiment labels (positive or negative).
# The text is vectorized and used to train the Perceptron model to associate words with positive/negative sentiment.

# For making predictions, new text input is vectorized using the same vocabulary. Then the Perceptron model makes a
# binary prediction on whether the new text has positive or negative sentiment.
# The output is based on whether the dot product of the input vector with the trained weight vectors is positive
# or negative.

# This provides a simple perceptron model for binary sentiment classification on textual data. The vectorization
# allows text to be converted into numerical features that the perceptron model can process. Overall,
# it demonstrates how a perceptron can be used for an NLP text classification task.
import numpy as np
from collections import defaultdict

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = defaultdict(float)
        self.bias = 0

    def fit(self, X, y):
        # Initialize the weights and bias
        self.weights = defaultdict(float)
        self.bias = 0

        for _ in range(self.n_iters):
            for i in range(len(X)):
                # Calculate the predicted sentiment (1 for positive, -1 for negative)
                pred = self.predict_sentiment(X[i])

                # Update weights and bias if prediction is incorrect
                if pred != y[i]:
                    self.update_weights(X[i], y[i], pred)

    def predict(self, X):
        predictions = [self.predict_sentiment(x) for x in X]
        return predictions

    def predict_sentiment(self, x):
        # Calculate the dot product of weights and features
        score = self.bias
        for word in x.split():
            if word in self.weights:
                score += self.weights[word]

        # Predict sentiment based on the dot product
        return 1 if score >= 0 else -1

    def update_weights(self, x, true_sentiment, predicted_sentiment):
        for word in x.split():
            self.weights[word] += self.lr * (true_sentiment - predicted_sentiment)

# Sample training data
X_train = [
    "I am excited about the perceptron network.",
    "We will not test the classifier with real data."
]
y_train = [1, -1, 1]

# Initialize and train the Perceptron model
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)

# Sample test data
X_test = ["I hated the movie, it was terrible!"]

# Make predictions
predictions = perceptron.predict(X_test)
print(predictions)

----------------------------------------------------------
#The following function is given
# F(x) = x1^2 + 2 x1 x2 + 2 x2^2 +x1
# use the steepest decent algorithm to find the minimum of the function.
# Plot the function in 3d and then plot the counter plot with the all the steps.
# use small value as a learning rate.
 ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Define the function F(x)
def F(x):
    x1, x2 = x
    return x1**2 + 2*x1*x2 + 2*x2**2 + x1

# Define the gradient of F(x)
def gradient_F(x):
    x1, x2 = x
    df_dx1 = 2*x1 + 2*x2 + 1
    df_dx2 = 2*x1 + 4*x2
    return np.array([df_dx1, df_dx2])

# Initialize parameters
learning_rate = 0.1  # Small learning rate
x = np.array([0.0, 0.0])  # Initial point
steps = [x]  # Record the steps

# Perform steepest descent
num_iterations = 50
for _ in range(num_iterations):
    x = x - learning_rate * gradient_F(x)
    steps.append(x)

# Convert the steps to a numpy array for plotting
steps = np.array(steps)

# Plot the 3D surface of F(x)
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = F([X1, X2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('F(x)')
ax.set_title('3D Surface of F(x)')

# Plot the contour plot with all the steps
plt.figure()
plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.plot(steps[:, 0], steps[:, 1], marker='o', markersize=5, color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot of F(x) with Steepest Descent Steps')
plt.grid()
plt.show()

----------------------------------------------------
# Follow the below instruction for writing the auto encoder code.

#The code implements a basic autoencoder model to learn word vector representations (word2vec style embeddings).
# It takes sentences of words as input and maps each word to an index in a vocabulary dictionary.

#The model has an encoder portion which converts word indexes into a low dimensional embedding via a learned weight
# matrix W1. This embedding is fed through anothe r weight matrix W2 to a hidden layer.

#The decoder portion maps the hidden representation back to the original word index space via weight matrix W3.

#The model is trained to reconstruct the original word indexes from the hidden embedding by minimizing the
# reconstruction loss using backpropagation.

#After training, the weight matrix W1 contains the word embeddings that map words in the vocabulary to dense
# vector representations. These learned embeddings encode semantic meaning and can be used as features for
# downstream NLP tasks.

import numpy as np

# Sample vocabulary and training data
vocab = {"apple": 0, "banana": 1, "cherry": 2, "date": 3}
sentences = [["apple", "banana"], ["cherry", "date"]]

# Define the dimensions of the word embeddings
embedding_dim = 3
learning_rate = 0.1
epochs = 100

# Create input data (convert words to indices)
X_train = []
for sentence in sentences:
    sentence_indices = [vocab[word] for word in sentence]
    X_train.extend(sentence_indices)

X_train = np.array(X_train)

# Initialize the weight matrices W1 and W3 with random values
W1 = np.random.rand(len(vocab), embedding_dim)
W3 = np.random.rand(embedding_dim, len(vocab))

# Training the autoencoder
for _ in range(epochs):
    # Encoder (embedding)
    Z = np.dot(X_train, W1)

    # Decoder
    X_pred = np.dot(Z, W3)

    # Calculate the reconstruction loss
    loss = np.mean((X_train - X_pred) ** 2)

    # Backpropagation to update weights
    gradient_W1 = -2 * np.dot((X_train - X_pred).reshape(-1, 1), Z.reshape(1, -1))
    gradient_W3 = -2 * np.dot(Z.reshape(-1, 1), (X_train - X_pred).reshape(1, -1))

    W1 -= learning_rate * gradient_W1
    W3 -= learning_rate * gradient_W3

    print(f"Epoch {_+1}, Loss: {loss}")

# Learned word embeddings (W1)
word_embeddings = W1

# Print the word embeddings
for word, index in vocab.items():
    print(f"Word: {word}, Embedding: {word_embeddings[index]}")