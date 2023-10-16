#============================Competition=======================================================================================================
# This dataset is about classify sample text to its associated class.
# The code have been used from chatgpt and stackoverflow
#**********************************
# Import Libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import re
import string



#**********************************
# Loading Dataset
train_filename = pd.read_csv('/home/ubuntu/NLP_AWS/Train.csv')

test_filename = pd.read_csv('/home/ubuntu/NLP_AWS/Test_submission.csv')


#**********************************


#**********************************

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
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


train_filename['Text'] = train_filename['Text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_filename['Text'], train_filename['Target'], test_size=0.2, random_state=42)

# Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = clf.predict(X_test_tfidf)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'F1 Score: {f1}')



# Preprocess the test data
test_filename['Text'] = test_filename['Text'].apply(preprocess_text)

# Vectorize the test data
X_test_data_tfidf = vectorizer.transform(test_filename['Text'])

# Predict labels for the test data
test_predictions = clf.predict(X_test_data_tfidf)

# Create a DataFrame with test predictions
test_filename['Target'] = test_predictions  # Assuming 'Target' is the column for predictions


print(f'F1 Score on Test Dataset: {f1}')
# Save the DataFrame with predictions to a CSV file
test_filename.to_csv('Test_submission_Bmariswamy.csv', index=False)