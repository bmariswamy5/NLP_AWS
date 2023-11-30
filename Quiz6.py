import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assuming your dataset has columns 'namn', 'antal', and 'gende'
# Load your dataset
# Example dataset structure:
# namn      antal   gende
# Emma      100     flicknamn
# Oskar     120     pojknamn

# Load the dataset
url = "path/to/your/dataset.csv"  # Replace with the path to your dataset
df = pd.read_csv(url)

# Preprocessing
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gende'])
names = df['namn'].tolist()
labels = df['gender'].tolist()

# Tokenize and pad sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(names)
num_chars = len(tokenizer.word_index) + 1
max_name_length = max(len(name) for name in names)
X = tokenizer.texts_to_sequences(names)
X_padded = pad_sequences(X, maxlen=max_name_length, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, labels, test_size=0.2, random_state=42)

# Build the model
embedding_dim = 10
model = Sequential()
model.add(Embedding(input_dim=num_chars, output_dim=embedding_dim, input_length=max_name_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict new names
new_names = ['Anna', 'Erik', 'Maria']
encoded_names = tokenizer.texts_to_sequences(new_names)
padded_names = pad_sequences(encoded_names, maxlen=max_name_length, padding='post')
predictions = model.predict(padded_names)
predicted_genders = ['Male' if pred >= 0.5 else 'Female' for pred in predictions]

# Display predictions
for name, gender in zip(new_names, predicted_genders):
    print(f"{name}: {gender}")
