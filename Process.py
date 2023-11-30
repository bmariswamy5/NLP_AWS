import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Read the CSV file into a DataFrame
# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('/home/ubuntu/NLP_AWS/legal_text_classification.csv')

# Drop rows with NaN values in the 'case_text_file' column
df = df.dropna(subset=['case_text'])


# Tokenize and preprocess the text
def preprocess(text):
    if pd.isna(text):  # Check for NaN values
        return []

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in STOP_WORDS]
    return tokens


# Apply the preprocessing to the 'case_text_file' column
df['processed_text'] = df['case_text'].apply(preprocess)

# Filter out rows where 'processed_text' is an empty list
df = df[df['processed_text'].apply(len) > 0]

# Create a Gensim dictionary
dictionary = Dictionary(df['processed_text'])

# Filter out extreme cases for efficient modeling
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create a bag-of-words representation
corpus = [dictionary.doc2bow(doc) for doc in df['processed_text']]

# Train LDA model
num_topics = 8  # Adjust the number of topics based on your requirements
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

# Print topics
topics = lda_model.print_topics(num_words=5)
for topic_idx, topic in enumerate(topics):
    print(f"Topic {topic_idx + 1}: {topic}")

# Assign topics to documents
document_topics = [lda_model[doc] for doc in corpus]

# Manually interpret topics and assign case types
case_types = {
    0: 'Land&Property',
    1: 'Financial',
    2: 'Criminal',
    3: 'Industrial&Labour',
    4: 'Motorvehicles',
    5: 'Constitution',
    6: 'Civil',
    7: 'Tax',
    8: 'Family',
    9: 'Environmental',
    10: 'Immigration',
    11: 'Human Rights',
    12: 'Intellectual Property',
    13: 'Healthcare',
    14: 'Insurance',
    15: 'Personal Injury',
    16: 'Employment',
    17: 'Contract',
    18: 'Estate Planning',
    19: 'Consumer Protection',
    20: 'Bankruptcy',
    21: 'Education',
    22: 'Securities',
    23: 'Antitrust',
    24: 'Defamation',
    25: 'Internet Law',
    26: 'Media Law',
    27: 'Arbitration',
    28: 'International Law',
    29: 'Real Estate',
    30: 'Sports Law',
    31: 'Other'
}

# Assign case types based on the dominant topic
df['case_types'] = [case_types[max(topics, key=lambda x: x[1])[0]] for topics in document_topics]

# Display the DataFrame with assigned case types
print(df[['case_text', 'case_types']])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, X_test, y_train, y_test = train_test_split(df['case_text'], df['case_type'], test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['case_type'].unique()))

# Tokenize and encode the training data
train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
train_labels = torch.tensor(y_train.astype('category').cat.codes.tolist())

# Tokenize and encode the test data
test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='pt')
test_labels = torch.tensor(y_test.astype('category').cat.codes.tolist())

# Create DataLoader for training and testing data
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_labels)
test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

# Convert predictions back to original labels
predicted_labels = pd.Categorical.from_codes(all_preds, categories=df['case_type'].unique())

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_labels)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test, predicted_labels))
