import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Ensure necessary NLTK packages are downloaded
nltk.download("punkt")
nltk.download("wordnet")

# Load intents
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
words, classes, documents = [], [], []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]))
classes = sorted(set(classes))

# Create training data
X_train, Y_train = [], []
for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    X_train.append(bag)
    Y_train.append(classes.index(doc[1]))

X_train, Y_train = np.array(X_train), np.array(Y_train)

# Build model
model = Sequential([
    Dense(128, input_shape=(len(X_train[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=200, batch_size=5, verbose=1)

# Save model and data
model.save("chatbot_model.h5")
with open("words.json", "w", encoding="utf-8") as file:
    json.dump(words, file)
with open("classes.json", "w", encoding="utf-8") as file:
    json.dump(classes, file)
