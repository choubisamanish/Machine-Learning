import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer # test into vectors
from tensorflow.keras.preprocessing.sequence import pad_sequences # vector into same length
from sklearn.model_selection import train_test_split

#Load the LSTM model
model = load_model('./LSTMRNN/next_word_lstm_rnn_model.h5')

with open('./LSTMRNN/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] #ensure the sequence length is max_sequence_len-1
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction with LSTM RNN")
input_text = st.text_input("Enter a sequence of words:")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted next word: {predicted_word}")
