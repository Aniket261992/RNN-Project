#Import the required packages

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the prediction model trained
model = load_model('simple_rnn_review_pred.h5')

#Load the IMDB data set word index

word_index = imdb.get_word_index()

#write method for preprocessing user input

def pre_processing(input_text):
    words = input_text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen = 500)
    return padded_review


#Create streamlit web app using the model
import streamlit as st
st.title('IMDB review sentiment prediction')
st.write('Please enter a review to classify:')

user_input = st.text_area('Movie review')

if st.button('Classify'):

    preprocessed_input=pre_processing(user_input)
    print(preprocessed_input)
    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')