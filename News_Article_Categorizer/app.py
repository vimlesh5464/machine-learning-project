import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Load Models and Vectorizers ---
bow_model = pickle.load(open("bow_model.pkl", "rb"))
tfidf_model = pickle.load(open("tfidf_model.pkl", "rb"))

bow_vectorizer = pickle.load(open("bow_vectorizer.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# --- Preprocessing Function ---
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Streamlit UI ---
st.title("BBC News Category Classifier")
st.write("This app classifies text into categories using BoW and TF-IDF models.")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Predict Category"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text!")
    else:
        processed = preprocess_text(user_input)

        # Vectorize input
        bow_vec = bow_vectorizer.transform([processed])
        tfidf_vec = tfidf_vectorizer.transform([processed])

        # Predictions
        bow_pred = bow_model.predict(bow_vec)[0]
        tfidf_pred = tfidf_model.predict(tfidf_vec)[0]

        st.subheader("Predicted Categories:")
        st.write(f"**BoW Prediction:** {bow_pred}")
        st.write(f"**TF-IDF Prediction:** {tfidf_pred}")

        st.success("Prediction complete!")
