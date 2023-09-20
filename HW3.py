# HW3
# Question 1
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# 1) Load the text file
with open('/home/ubuntu/NLP_AWS/Moby.txt','r',encoding='utf-8') as f:
   moby_text= f.read()

# 2) Tokenize the text into words.Number of token(words and punctuation symbols)
text_words = word_tokenize(moby_text)
number_of_tokens=len(text_words)
print("The total number of tokens in the moby text are:",number_of_tokens)

# 3)Unique tokens(unique words and punctuation)
number_of_unique_tokens=len(set(text_words))
print("The total number of unique tokens in the moby text are:",number_of_unique_tokens)

# 4) After lemmatizing what is the unique token count
lemmatizing=WordNetLemmatizer()
unique_lemmatized_tokens=len(set([lemmatizing.lemmatize(word, pos='v') for word in text_words]))
print("The total number of tokens after lemmatization the moby text are:",unique_lemmatized_tokens)

# 5) lexical diversity
lexical_diversity = float(unique_lemmatized_tokens) / float(number_of_tokens)
print("The lexical diversity is: ", lexical_diversity * 100, "%")

# 6) Percentage of 'Whale' or 'whale'
freq_dist=FreqDist(text_words)
whale_freq = freq_dist['whale'] + freq_dist['Whale']
print("The frequency distribution of whale is: ",(whale_freq / number_of_tokens) * 100 )

# 7) 20 most frequently occurring unique token in the frequency
print("The most frequently occurring unique token is :",freq_dist.most_common(20))

# 8) Tokens having length greater than 6 and frequency more than 160
print("the Tokens having length greater than 6 and frequency more than 160:",[token for token, freq in freq_dist.items() if len(token) > 6 and freq > 160])

# 9) Longest word in the text and its length
print("The longest word and its length from the text is:",max(text_words), ', ',len(max(text_words)))


# 10) unique word of frequency > 2000 and their frequency
print("unique words of token frequency >200:",{token: freq for token, freq in freq_dist.items() if freq > 2000})

# 11)Average number of token per sentence
sentences = sent_tokenize(moby_text)
print("The average number of token per sentence are:",number_of_tokens / len(sentences))

# 12) 5 most Frequent parts of speech and their frequency
pos_tags = nltk.pos_tag(text_words)
pos_freq_dist = FreqDist(tag for word, tag in pos_tags)
print("Frequent part of speech:",pos_freq_dist.most_common(5))
