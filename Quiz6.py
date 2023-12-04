import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
# Assuming you have a dataset with columns 'Name' and 'Gender'
# Replace this with your actual dataset
data = pd.read_csv('/home/ubuntu/NLP_AWS/Name_Data_set.csv')

# Convert gender labels to numerical values
label_encoder = LabelEncoder()
data['Gender_Label'] = label_encoder.fit_transform(data['gender'])

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define a simple Char-level embedding model
class CharEmbeddingModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size):
        super(CharEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Create a custom dataset class
class NameDataset(Dataset):
    def __init__(self, names, labels, char_to_index):
        self.names = names
        self.labels = labels
        self.char_to_index = char_to_index

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names.iloc[idx]['namn'].lower()  # Convert to lowercase
        label = self.labels.iloc[idx]['Gender_Label']
        name_indices = [self.char_to_index[char] for char in name if char in self.char_to_index]

        # Pad sequences to a fixed length (e.g., 10)
        padding_length = 10
        if len(name_indices) < padding_length:
            name_indices += [0] * (padding_length - len(name_indices))
        else:
            name_indices = name_indices[:padding_length]

        return torch.tensor(name_indices), torch.tensor(label)

# Create a character to index mapping
all_chars = set(' '.join(data['namn'].tolist()))
char_to_index = {char: i + 1 for i, char in enumerate(all_chars)}  # Reserve 0 for padding

# Initialize the model, loss function, and optimizer
input_size = len(char_to_index) + 1  # Add 1 for padding
embedding_dim = 200
hidden_size = 256
output_size = 1  # Binary classification
model = CharEmbeddingModel(input_size, embedding_dim,hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 30
batch_size = 3

train_dataset = NameDataset(train_data, train_data, char_to_index)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for names, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(names)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# Evaluate the model
model.eval()
test_dataset = NameDataset(test_data, test_data, char_to_index)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for names, labels in test_loader:
        outputs = model(names)
        predicted = torch.round(torch.sigmoid(outputs)).squeeze()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')


#____________________________________________________________
def predict_gender(model, name, char_to_index):
    model.eval()

    name_indices = [char_to_index[char] for char in name if char in char_to_index]

    # Pad sequences to a fixed length
    padding_length = 10
    if len(name_indices) < padding_length:
        name_indices += [0] * (padding_length - len(name_indices))
    else:
        name_indices = name_indices[:padding_length]

    with torch.no_grad():
        name_tensor = torch.tensor(name_indices).unsqueeze(0)
        output = model(name_tensor)
        prediction = torch.sigmoid(output).item()

    gender = "pojknamn" if prediction > 0.5 else "flicknamn"
    return gender

# Replace these names with your own samples
sample_names = ["Leela", "Tom", "Jerry", "Revanth"]

for name in sample_names:
    gender_prediction = predict_gender(model, name.lower(), char_to_index)
    print(f"Name: {name}, Predicted Gender: {gender_prediction}")