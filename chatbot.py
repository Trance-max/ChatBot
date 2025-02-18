import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

import streamlit as st

import json
import random
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Load data
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model("chatbot_model.h5")

with open("words.json", "r", encoding="utf-8") as file:
    words = json.load(file)
with open("classes.json", "r", encoding="utf-8") as file:
    classes = json.load(file)

# Preprocessing
def clean_text(text):
    words_list = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in words_list]

def bow(sentence):
    sentence_words = clean_text(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_data = bow(sentence)
    res = model.predict(np.array([bow_data]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

# Streamlit UI
st.title("💖 Love Chatbot 💬")
st.write("Talk to a chatbot that understands love and romance! ❤️")

user_input = st.text_input("You:")
if user_input:
    tag = predict_class(user_input)
    response = get_response(tag)
    st.text_area("Bot:", response, height=100)
