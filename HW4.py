#Package
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
nlp_large = spacy.load('en_core_web_lg')
#-----------------------------------------
# 1) Load the data using pandas

data = pd.read_csv("/home/ubuntu/NLP_AWS/data.csv")
word_attributes_list= []
entity_list = []
noun_phrases_list = []
grammatical_structure_list = []
similar_words_list = []

for title_text in data['uuid']:
    # Process the text using SpaCy
    doc = nlp(title_text)
    doc_large = nlp_large(title_text)
    # 2)word attributes
    word_attributes = pd.DataFrame({
        'Tokenized word': [token.text for token in doc],
        'StartIndex': [token.idx for token in doc],
        'Lemma': [token.lemma_ for token in doc],
        'Punctuation': [token.is_punct for token in doc],
        'White space': [token.is_space for token in doc],
        'WordShape': [token.shape_ for token in doc],
        'PartOfSpeech': [token.pos_ for token in doc],
        'POSTag': [token.tag_ for token in doc]
    })
    word_attributes_list.append(word_attributes)
    # 3)  entities
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    entity_list.append(entities)

    # 4)noun phrases
    noun_phrases_info = []
    for chunk in doc.noun_chunks:
        noun_phrases_info.append({
            'Noun Phrase': chunk.text,
            'Label': chunk.root.dep_,
            'Root': chunk.root.text
        })
    noun_phrases_list.append(pd.DataFrame(noun_phrases_info))

    # 5) Grammatical structure
    grammatical_structure_info = []
    for token in doc:
        grammatical_structure_info.append({
            'Text': token.text,
            'Dependency': token.dep_,
            'Head Text': token.head.text,
            'Head POS': token.head.pos_,
            'Children': [child.text for child in token.children]
        })
    grammatical_structure_list.append(pd.DataFrame(grammatical_structure_info))

    # 6)similarity
    target_word = "example"  # Change this to your desired target word
    similar_words = [word.text for word in nlp_large.vocab if
                     word.has_vector and word.is_lower and word.text != target_word and doc_large.similarity(
                         nlp_large(word.text)) > 0.5]
    similar_words_list.append(similar_words)

print("Word-Level Attributions:")
for word_attributes in word_attributes_list:
    print(word_attributes)

print("\nNamed Entities:")
for entities in entity_list:
    print(entities)

print("\nNoun Phrases:")
for noun_phrases_info in noun_phrases_list:
    print(noun_phrases_info)

print("\nGrammatical Structure Analysis:")
for grammatical_structure_info in grammatical_structure_list:
    print(grammatical_structure_info)

print("\nSimilar Words:")
for similar_words in similar_words_list:
    print(similar_words)



#------------ Question 1 end -------------------------------------------

# 1) Load data file
data1 = pd.read_csv('/home/ubuntu/NLP_AWS/data1.csv')

# 2) Plain text entities

sample_tweet = data1['text'].iloc[0]
doc = nlp(sample_tweet)

entities = []

# Extract entities and their explanations
for ent in doc.ents:
    entities.append([ent.text, ent.label_, spacy.explain(ent.label_)])

# Create a DataFrame with entity information
entity_columns = ['Entity Text', 'Entity Label', 'Entity Explanation']
entity_df = pd.DataFrame(entities, columns=entity_columns)

print(entity_df)

for idx, tweet in enumerate(data1['text']):
    doc = nlp(tweet)

    # Create a list to store redacted names
    redacted_names = []

    # Check if the tweet contains a name PERSON entity
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Redact the name by replacing it with "[REDACTED]"
            redacted_names.append(ent.text)

    # If names are found in the tweet, redact them
    if redacted_names:
        redacted_tweet = tweet
        for name in redacted_names:
            redacted_tweet = redacted_tweet.replace(name, "[REDACTED]")

        # Print the redacted tweet
        print(f"Redacted Tweet {idx + 1}: {redacted_tweet}")

#--------------Question 2 end------------------------------------------------------


sentence = "It was a good thing they were going home tomorrow."
doc = nlp(sentence)
# 1)Apply POS on sentence
for token in doc:
    print(f"Token: {token.text}, POS Tag: {token.pos_}")

# 2)Apply syntactic dependencies on same sentence
for token in doc:
    print(f"Token: {token.text}, Syntactic Dependency: {token.dep_}")

# 3) Apply named entities on provided senetence

sentence1 = "Apple is looking at buying U.K. startup for 1 billion dollar"
doc1=nlp(sentence1)
for ent in doc1.ents:
    print(f"Entity Text: {ent.text}, Entity Label: {ent.label_}, Entity Explanation: {spacy.explain(ent.label_)}")

# 4) Apply document similarity on two sentence
doc1 = nlp("The quick brown fox jumps over the lazy dog.")
doc2 = nlp("A brown fox quickly jumps over the lazy dog.")
print( doc1.similarity(doc2))
