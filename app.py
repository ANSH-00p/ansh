import os
import pickle
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

import streamlit as st

# Use os.path to make it work on Render
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'trained_modle.pkl')
vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
tfidf = pickle.load(open(vectorizer_path, 'rb'))
ps = PorterStemmer()

def data_cleaner(text):
    # Agar text string nahi hai (jaise NaN/Null), to use waise hi chhod dein
    if not isinstance(text, str):
        return []

    # Lowercase karna
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Alphanumeric check, Stopwords removal, Punctuation removal aur Stemming ek saath:
    clean_tokens = []
    stop_words = set(stopwords.words('english'))

    for i in tokens:
        if i.isalnum() and (i not in stop_words) and (i not in string.punctuation):
            clean_tokens.append(ps.stem(i))

    return " ".join(clean_tokens)

st.title("Email/SMS Classifier")
txt = st.text_area('Enter the message')
st.write(f"You wrote {len(txt)} characters.")
if st.button("Predict", type="primary"):
    transform_sms = data_cleaner(txt)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)
    if result == 1:
        st.header('SPAM ⚠️')
    else:
        st.header('NOT SPAM ✓')
