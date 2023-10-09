#package
import nltk
from nltk.corpus import nps_chat
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nps_chat_corpus = nps_chat.xml_posts()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Initialize a list to store cleaned and summarized documents
cleaned_and_summarized_docs = []


# Define a function for text preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Tokenization and cleaning
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

    # Join cleaned tokens back into a document
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

for post in nps_chat_corpus:
    text = post.text

    # Tokenization and cleaning
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

    # Join cleaned tokens back into a document
    cleaned_text = ' '.join(cleaned_tokens)

    cleaned_and_summarized_docs.append(cleaned_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_and_summarized_docs)

for i in range(len(cleaned_and_summarized_docs)):
    print("Document", i + 1, "Summary:")
    print(cleaned_and_summarized_docs[i])
    print("TF-IDF Representation:")
    print(tfidf_matrix[i])
    print("=" * 50)
# Compute cosine similarity between documents
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Print cosine similarity matrix
print(tfidf_matrix )
print(cosine_similarities)