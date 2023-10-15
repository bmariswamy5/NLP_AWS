import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
data = pd.read_csv('/home/ubuntu/NLP_AWS/data2.csv)

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







