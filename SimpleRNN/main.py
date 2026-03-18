import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential, load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('./SimpleRNN/simple_rnn_imdb_model.h5')

# step 2: Helper function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


#function to preprocess the user input review
def preprocess_text( text ):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## streamlit app
st.title( "IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")
user_input = st.text_area("Movie Review", "Type your review here...")

if st.button("Predict Sentiment"):
    prediction = model.predict(preprocess_text(user_input))
    #Make Prediction and display result
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Please enter a review and click the button to predict its sentiment.")