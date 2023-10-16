import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download NLTK resources for stop words and lemmatization
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load the training dataset
train_data = pd.read_csv('/home/ubuntu/NLP_AWS/Train.csv')


# Data Preprocessing
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Expand contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # Remove special characters and punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Tokenization (split the text into words)
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


train_data['Text'] = train_data['Text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train = train_data['Text']
y_train = train_data['Target']

# Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Load the test dataset
test_data = pd.read_csv('/home/ubuntu/NLP_AWS/Test_submission.csv')

# Preprocess the test data
test_data['Text'] = test_data['Text'].apply(preprocess_text)

# Vectorize the test data
X_test_tfidf = vectorizer.transform(test_data['Text'])

# Predict labels for the test data
test_predictions = clf.predict(X_test_tfidf)

# Deliverable 1: Training code
# This code trains the model.
# Note: The code for preprocessing and model training are combined for simplicity.

# Deliverable 2: Print F1 score on the test dataset
f1 = f1_score(test_data['Target'], test_predictions, average='weighted')
print(f'Deliverable 2 - F1 Score on Test Dataset: {f1}')

# Deliverable 3: Create a CSV file with labels
test_data['Target'] = test_predictions
test_data.to_csv('Test_submission_a.csv', index=False)

# Deliverable 4: Training subroutine (No separate subroutine in this code)
