import pickle
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem.porter import  PorterStemmer
from nltk.corpus import stopwords
import string

import streamlit as st

modle=pickle.load(open('trained_modle.pkl','rb'))
tfidf=pickle.load(open('vectorizer.pkl','rb'))
ps=PorterStemmer()

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
            # pd.stem(i) ki jagah ps.stem(i) use karein
            clean_tokens.append(ps.stem(i))

    return " ".join(clean_tokens)

st.title("Email/sms classifire")
txt = st.text_area('enter the massage')
st.write(f"You wrote {len(txt)} characters.")
if st.button("Pridict", type="primary"):
   transform_sms=data_cleaner(txt)
   vector_input=tfidf.transform([transform_sms])
   result=modle.predict(vector_input)
   if result==1:
       st.header('spam')
   else:
       st.header('not spam')
