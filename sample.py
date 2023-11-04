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
