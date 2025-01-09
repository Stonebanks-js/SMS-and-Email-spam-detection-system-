import nltk
import os
nltk.data.path.append('/home/aradhya_chaudhary/PycharmProjects/sms-spam-classifier/.venv/nltk_data')
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()


# Text transformation function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    # Remove non-alphanumeric characters and stopwords, then apply stemming
    text = [ps.stem(i) for i in text if
            i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(text)


# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Error: One or more required files (vectorizer.pkl, model.pkl) are missing!")
    raise e

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input field for SMS message
input_sms = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):
    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    st.write("Transformed Text:", transformed_sms)  # Debugging step

    # 2. Vectorize the input text
    vector_input = tfidf.transform([transformed_sms])
    st.write("Vectorized Input Shape:", vector_input.shape)  # Debugging step

    # 3. Predict the result using the model
    result = model.predict(vector_input)[0]
    st.write("Prediction Result:", result)  # Debugging step

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
