# Packages
import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# ---------------------------------------------------------------------------------------------------------------------
url = "https://en.wikipedia.org/wiki/Natural_language_processing"
response = requests.get(url)
html = BeautifulSoup(response.text, "html.parser")
text = " ".join([p.get_text() for p in html.find_all("p")])

def clean_text(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    cleaned_text = " ".join(cleaned_words)
    return cleaned_text

cleaned_text = clean_text(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_text)

filtered_text = remove_stopwords(cleaned_text)

sentences = sent_tokenize(filtered_text)


def extract_sentence_feature(sentence):
    words = word_tokenize(sentence)
    word_count = len(words)
    lemmatizer = WordNetLemmatizer()
    tagged_words = nltk.pos_tag([lemmatizer.lemmatize(word.lower()) for word in words])
    verb_count = len([word for word, pos in tagged_words if pos.startswith('V')])
    noun_count = len([word for word, pos in tagged_words if pos.startswith('N')])
    return word_count, verb_count, noun_count

sentence_features = [extract_sentence_feature(sentence) for sentence in sentences]

df = pd.DataFrame(sentence_features, columns=["Word Count", "Verb Count", "Noun Count"])
df["Sentence"] = sentences
print(df.head())

