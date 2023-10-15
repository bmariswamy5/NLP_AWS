import pandas as pd
import numpy as np
import random
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#----------------------------------------------
#Class_Ex1:
#  Use the following dataframe as the sample data.
# Find the conditional probability of Char given the Occurrence.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')
df = pd.DataFrame(
    {'Char': ['f', 'b', 'f', 'b', 'f', 'b', 'f', 'f'], 'Occurance': ['o1', 'o1', 'o2', 'o3', 'o2', 'o2', 'o1', 'o3'],
     'C': np.random.randn(8), 'D': np.random.randn(8)})

# Calculate the joint and marginal probabilities
joint_probabilities = df.groupby(['Char', 'Occurance']).size() / len(df)
marginal_probabilities = df.groupby('Occurance').size() / len(df)

# Calculate conditional probabilities
print(joint_probabilities / marginal_probabilities)
print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Use the following dataframe as the sample data.
# Find the conditional probability occurrence of thw word given a sentiment.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

df1 = pd.DataFrame({'Word': ['Good', 'Bad', 'Awesome', 'Beautiful', 'Terrible', 'Horrible'],
                    'Occurrence': ['One', 'Two', 'One', 'Three', 'One', 'Two'],
                    'sentiment': ['P', 'N', 'P', 'P', 'N', 'N'], })


print(df1.groupby(['Word', 'sentiment', 'Occurrence']).size() / df1.groupby(['sentiment', 'Occurrence']).size())

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Naive bayes and look at appropriate evaluation metric.
# 4- Explain your results very carefully.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')
# Step 1: Load and preprocess the data
data = pd.read_csv('/home/ubuntu/NLP_AWS/data2.csv',encoding='latin1')

# Assuming preprocessing includes lowercase conversion, you can do:
data['text'] = data['text'].str.lower()

# Step 2: Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Step 5: Evaluate the model's performance
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Step 6: Explain the results
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)
print(20 * '-' + 'End Q3' + 20 * '-')

# =================================================================
# Class_Ex4:
# Use Naive bayes classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

# Download the movie_reviews dataset if not already available
nltk.download('movie_reviews')

# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Define a function to extract features from text
def document_features(document):
    return ' '.join(document)

# Create feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the data into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Create a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB()),
])

# Define a parameter grid for grid search
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    'tfidf__max_df': [0.5, 0.75, 1.0],  # Maximum document frequency for features
    'classifier__alpha': [0.1, 1.0, 10.0],  # Smoothing parameter for Naive Bayes
}

# Perform grid search for parameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit([d for d, c in train_set], [c for d, c in train_set])

# Find the best parameters from the grid search
best_params = grid_search.best_params_

# Train the model with the best parameters on the entire training set
pipeline.set_params(**best_params)
pipeline.fit([d for d, c in train_set], [c for d, c in train_set])

# Evaluate the model on the test set
test_set_features = [(d, c) for (d, c) in test_set]
predicted = pipeline.predict([d for d, _ in test_set_features])
actual = [c for _, c in test_set_features]

# Print the classification report
report = classification_report(actual, predicted)
print(report)

# Print the best parameters found by grid search
print("Best Parameters:", best_params)



print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================

def calculate_accuracy(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    total = len(actual)
    accuracy = (correct / total) * 100
    return accuracy

def create_confusion_matrix(actual, predicted, labels):
    confusion_matrix = {}
    for a, p in zip(actual, predicted):
        if a not in confusion_matrix:
            confusion_matrix[a] = {}
        if p not in confusion_matrix[a]:
            confusion_matrix[a][p] = 0
        confusion_matrix[a][p] += 1
    return confusion_matrix

# Example usage
actual_labels = ["Positive", "Negative", "Positive", "Negative", "Neutral"]
predicted_labels = ["Positive", "Negative", "Positive", "Negative", "Positive"]

accuracy = calculate_accuracy(actual_labels, predicted_labels)
print("Accuracy: {:.2f}%".format(accuracy))

labels = ["Positive", "Negative", "Neutral"]
confusion_matrix = create_confusion_matrix(actual_labels, predicted_labels, labels)

# Print accuracy and confusion matrix
print("Accuracy: {:.2f}%".format(accuracy))
print("Confusion Matrix:")
for actual_label in labels:
    for predicted_label in labels:
        count = confusion_matrix.get(actual_label, {}).get(predicted_label, 0)
        print(f"{actual_label} (Actual) -> {predicted_label} (Predicted): {count} times")

# =================================================================
# Class_Ex6:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Logistic Regression  and look at appropriate evaluation metric.
# 4- Apply LSA method and compare results.
# 5- Explain your results very carefully.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')
# Step 1: Load and preprocess the data
data = pd.read_csv('/home/ubuntu/NLP_AWS/data2.csv',encoding='latin1')
# Perform text preprocessing, including normalization, cleaning, and stopwords removal.

# Step 2: Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Logistic Regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Step 5: Evaluate the Logistic Regression model
y_pred = logistic_regression.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Step 6: Apply Latent Semantic Analysis (LSA)
lsa = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
X_lsa = lsa.fit_transform(X)

# Step 7: Split the LSA-transformed data into training and testing sets
X_train_lsa, X_test_lsa, y_train, y_test = train_test_split(X_lsa, y, test_size=0.2, random_state=42)

# Step 8: Train a Logistic Regression model on LSA-transformed data
logistic_regression_lsa = LogisticRegression()
logistic_regression_lsa.fit(X_train_lsa, y_train)

# Step 5: Evaluate the Logistic Regression model on LSA-transformed data
y_pred_lsa = logistic_regression_lsa.predict(X_test_lsa)
accuracy_lsa = accuracy_score(y_test, y_pred_lsa)
report_lsa = classification_report(y_test, y_pred_lsa)
confusion_lsa = confusion_matrix(y_test, y_pred_lsa)

# Print the results for both approaches
print("Logistic Regression Results:")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

print("\nLSA Results:")
print("Accuracy (LSA):", accuracy_lsa)
print("Classification Report (LSA):\n", report_lsa)
print("Confusion Matrix (LSA):\n", confusion_lsa)
print(20 * '-' + 'End Q6' + 20 * '-')

# =================================================================
#
# Class_Ex7:
# Use logistic regression classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n-gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

nltk.download('movie_reviews')
# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Define a function to extract features from text
def document_features(document):
    return ' '.join(document)

# Create feature sets
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the data into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Extract features from the dataset
X_train = [d for d, c in train_set]
y_train = [c for d, c in train_set]
X_test = [d for d, c in test_set]
y_test = [c for d, c in test_set]

# Define a pipeline with TfidfVectorizer and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Define a parameter grid for grid search
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    'classifier__C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'classifier__penalty': ['l1', 'l2']  # Regularization penalty
}

# Perform grid search for parameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Find the best parameters from the grid search
best_params = grid_search.best_params_

# Train the model with the best parameters
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)

# Print the classification report and accuracy
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(report)
print("Accuracy:", accuracy)

print(20 * '-' + 'End Q7' + 20 * '-')
