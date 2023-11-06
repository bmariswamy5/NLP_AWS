import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


vocab_size = 10000
embedding_dim = 100

# Create a simple autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, 256),  # You can adjust the size of the hidden layer
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create the autoencoder model
autoencoder = Autoencoder(vocab_size, embedding_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training your autoencoder with some sample data

x_train = torch.randint(0, vocab_size, (100,)).long()  # Corrected size to match 100 samples

for epoch in range(10):
    optimizer.zero_grad()
    outputs = autoencoder(x_train)
    loss = criterion(outputs, autoencoder.encoder[0].weight[x_train])  # Use the model's own encoder for the target
    loss.backward()
    optimizer.step()

# After training, you can extract the word embeddings from the encoder
word_embeddings = autoencoder.encoder[0].weight.data
