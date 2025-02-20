import json
import random
import numpy as np
import tensorflow as tf
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer

# Load required data
nltk.download("punkt")
lemmatizer = WordNetLemmatizer()

with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

words = json.load(open("words.json", encoding="utf-8"))
classes = json.load(open("classes.json", encoding="utf-8"))
model = tf.keras.models.load_model("chatbot_model.h5")

# Function to preprocess input
def clean_text(text):
    word_list = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word.lower()) for word in word_list]

def bow(sentence):
    sentence_words = clean_text(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow_data = bow(sentence)
    res = model.predict(np.array([bow_data]))[0]
    return classes[np.argmax(res)]

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry love, puriyala! 🥺"

# Streamlit UI
st.title("💖 Tanglish Love Chatbot")
st.write("Romantic AI for Tamil love chats! 😍")

user_input = st.text_input("Talk to me baby! 😘")
if st.button("Send"):
    if user_input:
        tag = predict_class(user_input)
        response = get_response(tag)
        st.write(f"❤️ {response}")
