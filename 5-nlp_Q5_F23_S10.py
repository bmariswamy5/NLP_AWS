# =================================================================
# Class_Ex1:
# Lets consider the 2 following sentences
# Sentence 1: I  am excited about the perceptron network.
# Sentence 2: we will not test the classifier with real data.
# Design your bag of words set and create your input set.
# Choose your BOW words that suits perceptron network.
# Design your classes that Sent 1 has positive sentiment and sent 2 has a negative sentiment.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')
from sklearn.feature_extraction.text import CountVectorizer

# Given sentences
sentences = [
    "I am excited about the perceptron network.",
    "We will not test the classifier with real data."
]

# Labels for sentiment
sentiment_labels = ["positive", "negative"]

# Create a Bag of Words (BOW) set
bow_set = ["excited", "perceptron", "network", "test", "classifier", "real", "data"]

# Initialize CountVectorizer with the BOW set
vectorizer = CountVectorizer(vocabulary=bow_set)

# Vectorize the sentences
X = vectorizer.transform(sentences)

# Design Classes (Sentiment Labels)
y = sentiment_labels

# Display the BOW representation (X) and corresponding sentiment labels (y)
for i in range(len(sentences)):
    print(f"Sentence: '{sentences[i]}'")
    print(f"BOW representation: {X[i].toarray()}")
    print(f"Sentiment: {y[i]}\n")
print(20 * '-' + 'End Q1' + 20 * '-')
# =================================================================
# Class_Ex2:
# Use the same data in Example 1 but instead of hard-lim use log sigmoid as transfer function.
# modify your code inorder to classify negative and positive sentences correctly.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# Given sentences
sentences = [
    "I am excited about the perceptron network.",
    "We will not test the classifier with real data."
]

# Labels for sentiment
sentiment_labels = ["positive", "negative"]

# Create a Bag of Words (BOW) set
bow_set = ["excited", "perceptron", "network", "test", "classifier", "real", "data"]

# Initialize CountVectorizer with the BOW set
vectorizer = CountVectorizer(vocabulary=bow_set)

# Vectorize the sentences
X = vectorizer.transform(sentences)

# Design Classes (Sentiment Labels)
y = sentiment_labels

# Create an MLP Classifier with the log-sigmoid activation function
clf = MLPClassifier(activation='logistic', max_iter=1000)

# Fit the classifier to the data
clf.fit(X, y)

# Test the classifier
test_sentences = [
    "The perceptron network is amazing.",
    "Real data is essential for testing classifiers."
]

# Vectorize the test sentences
X_test = vectorizer.transform(test_sentences)

# Predict sentiment labels for the test sentences
predicted_labels = clf.predict(X_test)

# Display the test sentences and predicted sentiment labels
for i in range(len(test_sentences)):
    print(f"Test Sentence: '{test_sentences[i]}'")
    print(f"Predicted Sentiment: {predicted_labels[i]}\n")


print(20 * '-' + 'End Q2' + 20 * '-')

# =================================================================
# Class_Ex3:
# The following function is given
# F(x) = x1^2 + 2 x1 x2 + 2 x2^2 +x1
# use the steepest decent algorithm to find the minimum of the function.
# Plot the function in 3d and then plot the counter plot with the all the steps.
# use small value as a learning rate.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function F(x)
def F(x):
    x1, x2 = x
    return x1**2 + 2*x1*x2 + 2*x2**2 + x1

# Calculate the gradient of F(x)
def gradient(x):
    x1, x2 = x
    df_dx1 = 2*x1 + 2*x2 + 1
    df_dx2 = 2*x1 + 4*x2
    return np.array([df_dx1, df_dx2])

# Initialize starting point and learning rate
x = np.array([2, 2])
learning_rate = 0.1
num_iterations = 100

# Lists to store the path of optimization
path = [x]

# Perform steepest descent optimization
for _ in range(num_iterations):
    x = x - learning_rate * gradient(x)
    path.append(x)

# Convert path to numpy array
path = np.array(path)

# Create a meshgrid for plotting
x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
z = F([x1, x2])

# Plot the function in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('F(x)')
ax.set_title('3D Plot of F(x)')

# Plot the contour plot with optimization path
plt.figure()
plt.contour(x1, x2, z, levels=20, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], marker='o', color='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour Plot of F(x) with Optimization Path')
plt.show()


print(20 * '-' + 'End Q3' + 20 * '-')

# =================================================================
# Class_Ex4:
# Use the following corpus of data
# sent1 : 'This is a sentence one, and I want to all data here.',
# sent2 :  'Natural language processing has nice tools for text mining and text classification.
#           I need to work hard and try a lot of exercises.',
# sent3 :  'Ohhhhhh what',
# sent4 :  'I am not sure what I am doing here.',
# sent5 :  'Neural Network is a power method. It is a very flexible architecture'

# Train ADALINE network to find  a relationship between POS (just verbs and nouns) and the the length of the sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Sample sentences
sentences = [
    'This is a sentence one, and I want to all data here.',
    'Natural language processing has nice tools for text mining and text classification. I need to work hard and try a lot of exercises.',
    'Ohhhhhh what',
    'I am not sure what I am doing here.',
    'Neural Network is a power method. It is a very flexible architecture'
]

# Initialize NLTK for POS tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Lists to store feature vectors and sentence lengths
feature_vectors = []
sentence_lengths = []

# POS tags of interest (verbs and nouns)
pos_tags_of_interest = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS']

for sentence in sentences:
    # Tokenize the sentence and get POS tags
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)

    # Count the verbs and nouns
    num_verbs_nouns = sum(1 for word, pos in pos_tags if pos in pos_tags_of_interest)

    # Calculate the sentence length
    sentence_length = len(words)

    # Create a feature vector [num_verbs_nouns, sentence_length]
    feature_vector = [num_verbs_nouns, sentence_length]

    # Append to the lists
    feature_vectors.append(feature_vector)
    sentence_lengths.append(sentence_length)

# Convert lists to numpy arrays
X = np.array(feature_vectors)
y = np.array(sentence_lengths)

# Initialize and train an ADALINE network
adline = SGDRegressor(eta0=0.01, max_iter=10000, tol=1e-3)
adline.fit(X, y)

# Predict sentence lengths
predicted_lengths = adline.predict(X)

# Calculate mean squared error
mse = mean_squared_error(y, predicted_lengths)

print("Trained ADALINE Network:")
print("Weights (Coefficients):", adline.coef_)
print("Intercept:", adline.intercept_)
print("Mean Squared Error (MSE):", mse)

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Read the dataset.csv file. This dataset is about the EmailSpam.
# Use a two layer network and to classify each email
# You are not allowed to use any NN packages.
# You can use previous NLP packages to read the data process it (NLTK, spaCY)
# Show the classification report and mse of training and testing.
# Try to improve your F1 score. Explain which methods you used.
# Hint. Clean the dataset, use all the preprocessing techniques that you learned.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')
# =================================================================
# Class_Ex7:
# The objective of this exercise to show the inner workings of Word2Vec in python using numpy.
# Do not be using any other libraries for that.
# We are not looking at efficient implementation, the purpose here is to understand the mechanism
# behind it. You can find the official paper here. https://arxiv.org/pdf/1301.3781.pdf
# The main component of your code should be the followings:
# Set your hyper-parameters
# Data Preparation (Read text file)
# Generate training data (indexing to an integer and the onehot encoding )
# Forward and backward steps of the autoencoder network
# Calculate the error
# look at error at by varying hidden dimensions and window size
#--------------------------------------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#--------------------------------------------------------------------------------------------
LR = 1e-2
N_EPOCHS = 2000
PRINT_LOSS_EVERY = 1000
EMBEDDING_DIM = 2
#--------------------------------------------------------------------------------------------
corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']
#--------------------------------------------------------------------------------------------
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results
#--------------------------------------------------------------------------------------------
corpus = remove_stop_words(corpus)
#--------------------------------------------------------------------------------------------
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = list(set(words))
print(words)
#--------------------------------------------------------------------------------------------
word2int = {word: i for i, word in enumerate(words)}

sentences = [sentence.split() for sentence in corpus]

WINDOW_SIZE = 2
data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

df = pd.DataFrame(data, columns=['input', 'label'])
print(df.head(10))

#--------------------------------------------------------------------------------------------
ONE_HOT_DIM = len(words)

def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding
#--------------------------------------------------------------------------------------------
X = np.array([to_one_hot_encoding(word2int[x]) for x in df['input']])
Y = np.array([to_one_hot_encoding(word2int[y]) for y in df['label']])
#--------------------------------------------------------------------------------------------
class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, lr):
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        self.W2 = np.random.randn(embedding_dim, vocab_size)
        self.lr = lr

    def train(self, X, Y, epochs, print_loss_every):
        for epoch in range(epochs):
            loss = 0
            for x, y in zip(X, Y):
                h = np.dot(self.W1.T, x)
                u = np.dot(self.W2.T, h)
                y_pred = 1 / (1 + np.exp(-u))

                error = y - y_pred
                loss += np.sum(error**2)

                grad_w2 = np.outer(h, error)
                grad_w1 = np.outer(x, np.dot(self.W2, error.T))

                self.W1 += self.lr * grad_w1
                self.W2 += self.lr * grad_w2

            if epoch % print_loss_every == 0:
                print(f"Epoch {epoch} | Loss {loss}")

        return self.W1

word2vec = Word2Vec(ONE_HOT_DIM, EMBEDDING_DIM, LR)
word_embeddings = word2vec.train(X, Y, N_EPOCHS, PRINT_LOSS_EVERY)

w2v_df = pd.DataFrame(word_embeddings, columns=['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
#--------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))

PADDING = 1.0
x_axis_min = np.amin(word_embeddings, axis=0)[0] - PADDING
y_axis_min = np.amin(word_embeddings, axis=0)[1] - PADDING
x_axis_max = np.amax(word_embeddings, axis=0)[0] + PADDING
y_axis_max = np.amax(word_embeddings, axis=0)[1] + PADDING

plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)
plt.show()
print(20 * '-' + 'End Q7' + 20 * '-')
#--------------------------------------------------------------------------------------------