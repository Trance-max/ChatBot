import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout # type: ignore

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load intents file (Fix: UTF-8 encoding issue)
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
words, classes, documents = [], [], []
ignore_letters = ["?", "!", ".", ","]

# Tokenize and process patterns
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)  # Tokenize sentence
        words.extend(word_list)  # Add words to vocabulary
        documents.append((word_list, intent["tag"]))  # Store tag association
        if intent["tag"] not in classes:
            classes.append(intent["tag"])  # Unique intent tags

# Lemmatize words and remove duplicates
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))

# Create training data
X_train, Y_train = [], []
for doc in documents:
    bag = [1 if lemmatizer.lemmatize(w.lower()) in [lemmatizer.lemmatize(x.lower()) for x in doc[0]] else 0 for w in words]
    X_train.append(bag)
    Y_train.append(classes.index(doc[1]))

X_train, Y_train = np.array(X_train), np.array(Y_train)

# Build the Neural Network model
model = Sequential([
    Dense(128, input_shape=(len(X_train[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")  # Output layer for classification
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, Y_train, epochs=200, batch_size=5, verbose=1)

# Save model and training data
model.save("chatbot_model.h5")
with open("words.json", "w", encoding="utf-8") as file:
    json.dump(words, file)
with open("classes.json", "w", encoding="utf-8") as file:
    json.dump(classes, file)

print("Training Complete! ✅ Model saved as 'chatbot_model.h5'")
