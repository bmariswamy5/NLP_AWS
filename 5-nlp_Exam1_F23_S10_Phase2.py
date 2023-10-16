#**********************************
import nltk
from nltk.corpus import movie_reviews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
#**********************************
#==================================================================================================================================================================
# Q1:
"""
For this question you need to download a text data from the NLTK movie_reviews.
Use you knowledge that you learned in the class and clean the text appropriately.
After Cleaning is done, please find the numerical representation of text by any methods that you learned.
You need to find a creative way to label the sentiment of the sentences.
This dataset already has positive and negative labels.
Labeling sentences as 'positive' or 'negative' based on sentiment scores and named then predicted sentiments.
Create a Pandas dataframe with sentences, true sentiment labels and predicted sentiment labels.
Calculate the accuracy of your predicted sentiment and true sentiments.
"""
#==================================================================================================================================================================

print(20*'-' + 'Begin Q1' + 20*'-')
nltk.download('movie_reviews')

# Load the movie reviews dataset
positive_reviews = [movie_reviews.raw(file_id) for file_id in movie_reviews.fileids('pos')]
negative_reviews = [movie_reviews.raw(file_id) for file_id in movie_reviews.fileids('neg')]

# Initialize the VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to get sentiment labels
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Get true sentiments and perform sentiment analysis
true_sentiments = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)
all_reviews = positive_reviews + negative_reviews
predicted_sentiments = [get_sentiment(review) for review in all_reviews]

# Create a Pandas DataFrame
df = pd.DataFrame({'Sentences': all_reviews, 'True Sentiment': true_sentiments, 'Predicted Sentiment': predicted_sentiments})

# View DataFrame and Calculate Accuracy
print(df)
accuracy = (df['True Sentiment'] == df['Predicted Sentiment']).mean()
print(f'Accuracy: {accuracy:.2%}')
print(20*'-' + 'End Q1' + 20*'-')


