import re
import streamlit as st
import pandas as pd
from transformers import pipeline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

def upload_file(file_name):
    # File uploader (for Streamlit)
    file = st.file_uploader(f"Choose the {file_name}", type=["txt"])

    if file is not None:
        # Read the content of the file as bytes
        content_bytes = file.read()

        if content_bytes:
            # Decode the bytes into a string
            content = content_bytes.decode('utf-8')

            return content
        else:
            st.error("File is empty. Please choose a file with content.")
            return None
    else:
        return None
def sidebar():
    with st.sidebar:
        genre = st.radio(
            "Choose your model",
            ["Sentiment Analysis BERT","LDA Model"],
            captions=["LDA", "Use logistic regression to predict."],
            index=None,
        )
        return genre


def analyze_sentiment_bert(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    st.write(f"Sentiment: {result['label']}")
    st.write(f"Sentiment Score: {result['score']}")

    # Return 'positive' or 'negative'
    return result['label'].lower()

def perform_lda(text):
    processed_text = preprocess_string(text)
    dictionary = corpora.Dictionary([processed_text])
    corpus = [dictionary.doc2bow(processed_text)]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary)

    # Get the topic distribution for the document
    topic_distribution = lda_model.get_document_topics(corpus[0])
    dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]

    # Display the dominant topic
    st.write(f"Dominant Topic: {dominant_topic}")
    return topic_distribution