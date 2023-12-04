import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


# Custom model class with additional layers
class CustomModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.lstm_layer = nn.LSTM(768, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.conv_layer = nn.Conv1d(768, 256, kernel_size=3, padding=1)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # LSTM Layer
        lstm_outputs, _ = self.lstm_layer(bert_outputs)
        lstm_outputs = lstm_outputs[:, -1, :]  # Take the output from the last time step
        # Convolution Layer
        conv_outputs = self.conv_layer(bert_outputs.transpose(1, 2)).max(dim=2)[0]

        return bert_outputs, lstm_outputs, conv_outputs


# Load your dataset (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('/home/ubuntu/NLP_AWS/Train_ML.csv')

# Assuming you have a 'Title' column and a column for each label
# Adjust these column names based on your dataset
text_column = 'TITLE'
label_columns = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

# Convert labels to binary (0 or 1)
for label in label_columns:
    data[label] = data[label].apply(lambda x: 1 if x > 0 else 0)

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the training data
train_inputs = tokenizer(list(train_data[text_column]), padding=True, truncation=True, return_tensors='pt',
                         max_length=256)
train_labels = torch.tensor(train_data[label_columns].values, dtype=torch.float32)

# Tokenize and encode the test data
test_inputs = tokenizer(list(test_data[text_column]), padding=True, truncation=True, return_tensors='pt',
                        max_length=256)
test_labels = torch.tensor(test_data[label_columns].values, dtype=torch.float32)

# Dataset and DataLoader
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model
num_labels = len(label_columns)
model = CustomModel(num_labels)

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
epochs = 3  # Adjust as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        inputs, masks, labels = batch
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        bert_outputs, lstm_outputs, conv_outputs = model(input_ids=inputs, attention_mask=masks)

        # You may want to adjust the loss calculation based on your specific use case
        loss = criterion(bert_outputs, labels) + criterion(lstm_outputs, labels) + criterion(conv_outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
all_preds_bert, all_preds_lstm, all_preds_conv = [], [], []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        inputs, masks, labels = batch
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        bert_outputs, lstm_outputs, conv_outputs = model(input_ids=inputs, attention_mask=masks)

        all_preds_bert.append(torch.sigmoid(bert_outputs).cpu().numpy())
        all_preds_lstm.append(torch.sigmoid(lstm_outputs).cpu().numpy())
        all_preds_conv.append(torch.sigmoid(conv_outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds_bert = np.concatenate(all_preds_bert, axis=0)
all_preds_lstm = np.concatenate(all_preds_lstm, axis=0)
all_preds_conv = np.concatenate(all_preds_conv, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Binary threshold for classification
threshold = 0.5
pred_labels_bert = (all_preds_bert > threshold).astype(int)
pred_labels_lstm = (all_preds_lstm > threshold).astype(int)
pred_labels_conv = (all_preds_conv > threshold).astype(int)

# Metrics
accuracy_bert = accuracy_score(all_labels, pred_labels_bert)
kappa_bert = cohen_kappa_score(all_labels.argmax(axis=1), pred_labels_bert.argmax(axis=1))
accuracy_lstm = accuracy_score(all_labels, pred_labels_lstm)
kappa_lstm = cohen_kappa_score(all_labels.argmax(axis=1), pred_labels_lstm.argmax(axis=1))
accuracy_conv = accuracy_score(all_labels, pred_labels_conv)
kappa_conv = cohen_kappa_score(all_labels.argmax(axis=1), pred_labels_conv.argmax(axis=1))

print(f'Bert Accuracy: {accuracy_bert:.4f}, Kappa Score: {kappa_bert:.4f}')
print(f'LSTM Accuracy: {accuracy_lstm:.4f}, Kappa Score: {kappa_lstm:.4f}')
print(f'Convolution Accuracy: {accuracy_conv:.4f}, Kappa Score: {kappa_conv:.4f}')

# Create a submission file
submission_df = pd.DataFrame({
    'BERT_CS': pred_labels_bert[:, 0],
    'BERT_Physics': pred_labels_bert[:, 1],
    'BERT_Mathematics': pred_labels_bert[:, 2],
    'BERT_Statistics': pred_labels_bert[:, 3],
    'BERT_Quantitative_Biology': pred_labels_bert[:, 4],
    'BERT_Quantitative_Finance': pred_labels_bert[:, 5],
    'LSTM_CS': pred_labels_lstm[:, 0],
    'LSTM_Physics': pred_labels_lstm[:, 1],
    'LSTM_Mathematics': pred_labels_lstm[:, 2],
    'LSTM_Statistics': pred_labels_lstm[:, 3],
    'LSTM_Quantitative_Biology': pred_labels_lstm[:, 4],
    'LSTM_Quantitative_Finance': pred_labels_lstm[:, 5],
    'Convolution_CS': pred_labels_conv[:, 0],
    'Convolution_Physics': pred_labels_conv[:, 1],
    'Convolution_Mathematics': pred_labels_conv[:, 2],
    'Convolution_Statistics': pred_labels_conv[:, 3],
    'Convolution_Quantitative_Biology': pred_labels_conv[:, 4],
    'Convolution_Quantitative_Finance': pred_labels_conv[:, 5],
})

submission_df.to_csv('/home/ubuntu/NLP_AWS/Test_submission_netid.csv', index=False)
print("Submission file created.")
