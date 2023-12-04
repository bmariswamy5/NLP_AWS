import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import numpy as np

# Load your dataset (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('/home/ubuntu/NLP_AWS/Train_ML.csv')

# Assuming you have a 'Title' column and a column for each label
# Adjust these column names based on your dataset
text_column = 'TITLE'
label_columns = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

# Model with LSTM layer
class BertLSTMModel(torch.nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, num_labels):
        super(BertLSTMModel, self).__init__()
        self.bert = bert_model
        self.lstm = torch.nn.LSTM(768, lstm_hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_input = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]  # Take the last output of the sequence
        logits = self.fc(lstm_output)
        return logits

model = BertLSTMModel(BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_columns)), lstm_hidden_size=128, num_labels=len(label_columns))

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
        outputs = model(input_ids=inputs, attention_mask=masks)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        inputs, masks, labels = batch
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        outputs = model(input_ids=inputs, attention_mask=masks)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Binary threshold for classification
threshold = 0.5
pred_labels = (all_preds > threshold).astype(int)

# Metrics
accuracy = accuracy_score(all_labels, pred_labels)
kappa = cohen_kappa_score(all_labels.argmax(axis=1), pred_labels.argmax(axis=1))

print(f'Accuracy: {accuracy:.4f}')
print(f'Cohen Kappa Score: {kappa:.4f}')

# Create a submission file
submission_df = pd.DataFrame(pred_labels, columns=label_columns)
submission_df.to_csv('/home/ubuntu/NLP_AWS/Test_submission_netid2.csv', index=False)
print("Submission file created.")
