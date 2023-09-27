#Package
import pandas as pd
import spacy


#-----------------------------------------
# 1) Load the data using pandas

data = pd.read_csv("data.csv")

# 2) to find word level attributes using spacy
nlp = spacy.load("en_core_web_sm")

title1 = data['Title'].iloc[0]
doc = nlp(title1)
word_attributes = []

for token in doc:
    word_attributes.append([token.text, token.idx, token.lemma_, token.is_punct, token.is_space, token.shape_, token.pos_, token.tag_])

columns = ['Tokenized word', 'StartIndex', 'Lemma', 'Punctuation', 'WhiteSpace', 'WordShape', 'PartOfSpeech', 'POSTag']
word_attributes_df = pd.DataFrame(word_attributes, columns=columns)

for ent in doc.ents:
    print(ent.text, ent.label_)

# 4)To find chunk the noun phrases, label and roots of chunk
title2=data['Title'].iloc[1]
doc= nlp(title1)
noun_chunk_phrase = list(doc.noun_chunks)
print( [(chunk.text, chunk.root.text) for chunk in noun_chunk_phrase])

# 5)SPacy to analyzes the grammatical structure of a sentence

grammatical_structure = []

for token in doc:
    grammatical_structure.append([token.text, token.dep_, token.head.text, token.head.pos_, [child.text for child in token.children]])


# 6) Use spacy to find word similarity measure
nlp_large = spacy.load('en_core_web_lg')
target_word = nlp_large('word_to_find_similar_words_for')
print([word.text for word in nlp_large.vocab if word.has_vector and word.is_lower and word.is_alpha and not word.is_stop and word.similarity(target_word) > 0.5])
