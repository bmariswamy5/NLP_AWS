import spacy
nlp = spacy.load("en_core_web_sm")
# -----------------------------------------------------------------------------------------
text = """SpaCy is a free, open-source library for advanced Natural Language Processing 
(NLP) in Python. If you're working with a lot of text, you'll eventually want to know 
more about it. For example, what's it about? What do the words mean in context? 
Who is doing what to whom? What companies and products are mentioned?
 Which texts are similar to each other?
SpaCy helps you answer these questions with a powerful, streamlined API that's easy 
to use and integrates seamlessly with other Python libraries. SpaCy also comes with 
pre-trained statistical models and word vectors, and supports deep learning workflows 
that allow you to train and update neural network models on your own data."""
# -----------------------------------------------------------------------------------------
# 1:
# Your code here
doc = nlp(text)

sentences = list(doc.sents)
print(f"Number of sentences: {len(sentences)}")
# -----------------------------------------------------------------------------------------
# 2
# Your code here
first_sentence= list(sentences[0])
print(f"Number of tokens in the first sentence: {len(first_sentence)}")



# -----------------------------------------------------------------------------------------
#3
# Your code here
print("{:<15} {:<10}".format("Token", "POS"))
for token in first_sentence:
    print("{:<15} {:<10}".format(token.text, token.pos_))



# -----------------------------------------------------------------------------------------
#4
# Your code here
adj = [token for token in doc if token.sentiment > 0]
print(f"Number of positive adjectives: {len(adj)}")



# -----------------------------------------------------------------------------------------
#5
# Your code here

for i, sentence in enumerate(doc.sents, 1):
    nouns = [token.text for token in sentence if token.pos_ == 'NOUN']
    num_nouns = len(nouns)
    print(f"Sentence {i}: {num_nouns} nouns")
# I have selected noun feature to count number of noun that occur in the  given sample text
