#Package
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
nlp_large = spacy.load('en_core_web_lg')
#-----------------------------------------
# 1) Load the data using pandas

data = pd.read_csv("/home/ubuntu/NLP_AWS/data.csv")



word_attributions_all = pd.DataFrame(columns=['Tokenized word', 'StartIndex', 'Lemma', 'Punctuation', 'Whitespace', 'WordShape', 'PartOfSpeech', 'POSTag'])
entities_all = pd.DataFrame(columns=['Entity', 'Label'])
noun_phrases_all = pd.DataFrame(columns=['Noun Phrase', 'Label', 'Root'])
grammatical_structure_all = pd.DataFrame(columns=['Text', 'Dependency', 'Head Text', 'Head POS', 'Children'])
similarity_scores = []

for index, row in data.iterrows():
    # Get the title from the current row
    texts = row['uuid']
    doc = nlp(texts)
    word_attributions = pd.DataFrame({
        'Tokenized word': [token.text for token in doc],
        'StartIndex': [token.idx for token in doc],
        'Lemma': [token.lemma_ for token in doc],
        'Punctuation': [token.is_punct for token in doc],
        'Whitespace': [token.is_space for token in doc],
        'WordShape': [token.shape_ for token in doc],
        'PartOfSpeech': [token.pos_ for token in doc],
        'POSTag': [token.tag_ for token in doc]
    })
    word_attributions_all = pd.concat([word_attributions_all, word_attributions])

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entities_df = pd.DataFrame(entities, columns=['Entity', 'Label'])
    entities_all = pd.concat([entities_all, entities_df])

print(word_attributions_all)
print(entities_all)


for index, row in data.iterrows():
    # Get the title from the current row
    text1 = row['site_url']

    # Tokenize the text
    doc1 = nlp(text1)
    doc_large = nlp_large(text1)
    # Initialize lists to store noun phrases, labels, and roots for the current title
    noun_phrases = []
    labels = []
    roots = []

    # Chunk noun phrases, label them, and find roots for the current title
    for chunk in doc1.noun_chunks:
        noun_phrases.append(chunk.text)
        labels.append(chunk.root.text)
        roots.append(chunk.root.head.text)

    # Create a DataFrame for the current title's results
    noun_phrases_df = pd.DataFrame({
        'Noun Phrase': noun_phrases,
        'Label': labels,
        'Root': roots
    })

    # Concatenate the results for the current title to the overall DataFrame
    noun_phrases_all = pd.concat([noun_phrases_all, noun_phrases_df])
    dependency_info = []
    for token in doc1:
        dependency_info.append(
            (token.text, token.dep_, token.head.text, token.head.pos_, [child.text for child in token.children]))
    grammatical_structure_df = pd.DataFrame(dependency_info,columns=['Text', 'Dependency', 'Head Text', 'Head POS', 'Children'])
    grammatical_structure_all = pd.concat([grammatical_structure_all, grammatical_structure_df])

    word1 = row['site_url']
    word2 = row['site_url']
    similarity = nlp_large(word1).similarity(nlp_large(word2))
    similarity_scores.append((word1, word2, similarity))
    similarity_df = pd.DataFrame(similarity_scores, columns=['word1', 'word2', 'Similarity'])


print(noun_phrases_all)
print(grammatical_structure_all)
print(similarity_df)

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
