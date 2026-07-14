# Import Libraries

import streamlit as st
import pandas as pd
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page Configuration

st.set_page_config(
    page_title="AI Fake News Detection System",
    layout="wide"
)

# Application Title

st.title("AI Fake News Detection System")

st.write(
    """
    This application predicts whether a news article is
    Fake News or Real News using Natural Language Processing
    and Machine Learning.
    """
)

# Load Saved Model

model = joblib.load("model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download Required NLTK Resources

nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLP Objects

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# Prediction

if analyze_button:

    if news_title.strip() == "" and news_content.strip() == "":

        st.warning("Please enter news title and news content.")

    else:

        complete_news = news_title + " " + news_content

        clean_text = preprocess_text(complete_news)

        vector = vectorizer.transform([clean_text])

        prediction = model.predict(vector)

        st.subheader("NLP Processing Steps")

        steps = [
            "Convert input text to lowercase",
            "Remove URLs",
            "Remove punctuation",
            "Remove numbers",
            "Tokenization",
            "Stopword Removal",
            "Lemmatization",
            "TF-IDF Feature Extraction"
        ]

        for step in steps:
            st.write(step)

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("Prediction : Fake News")
        else:
            st.success("Prediction : Real News")
            
# Text Preprocessing Function

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

sample_option = st.selectbox(
      
    "Choose Sample News",
    [
        "Custom News",
        "Real News Sample",
        "Fake News Sample"
    ]
)

# Default Sample Values

sample_title = ""
sample_content = ""

# Load Sample News Based on Selection

if sample_option == "Real News Sample":

    sample_title = "Reuters reports US inflation slows"

    sample_content = "Reuters reported that inflation in the United States slowed during the latest quarter according to official government statistics."

elif sample_option == "Fake News Sample":

    sample_title = "Breaking: Secret government plan exposed"

    sample_content = "A viral social media post claims that the government has secretly banned all cash transactions from tomorrow. No official source has confirmed this claim."

# User News Input

news_title = st.text_input(
    "Enter News Title",
    value=sample_title
)

news_content = st.text_area(
    "Enter News Content",
    value=sample_content,
    height=250
)

# Analyze News Button

analyze_button = st.button("Analyze News")

# Sidebar

st.sidebar.title("Project Information")

st.sidebar.write("Project : AI Fake News Detection System")
st.sidebar.write("Technology : NLP and Machine Learning")
st.sidebar.write("Algorithm : Linear Support Vector Classifier")
st.sidebar.write("Feature Extraction : TF-IDF Vectorization")
st.sidebar.write("Framework : Streamlit")

st.sidebar.header("About Project")

st.sidebar.write(
    """
AI Fake News Detection System is developed using
Natural Language Processing and Machine Learning.

The application predicts whether the given news
is Fake News or Real News.
"""
)

st.sidebar.header("Technologies Used")

st.sidebar.write("- Python")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Natural Language Processing")
st.sidebar.write("- TF-IDF Vectorization")
st.sidebar.write("- Scikit-Learn")
st.sidebar.write("- LinearSVC")

st.sidebar.header("Project Workflow")

st.sidebar.write("1. User Input")
st.sidebar.write("2. Text Preprocessing")
st.sidebar.write("3. TF-IDF Vectorization")
st.sidebar.write("4. Machine Learning Prediction")
st.sidebar.write("5. Display Result")

# Sidebar Information

st.sidebar.header("About Project")

st.sidebar.write(
    """
    AI Fake News Detection System is developed
    using Natural Language Processing and
    Machine Learning.

    The application predicts whether
    the given news is Fake or Real.
    """
)

st.sidebar.header("Technologies Used")

st.sidebar.write("- Python")

st.sidebar.write("- Streamlit")

st.sidebar.write("- Natural Language Processing")

st.sidebar.write("- TF-IDF Vectorization")

st.sidebar.write("- Scikit-Learn")

st.sidebar.write("- Linear Support Vector Classifier")


st.sidebar.header("Project Workflow")

st.sidebar.write("User Input")

st.sidebar.write("Text Preprocessing")

st.sidebar.write("TF-IDF Vectorization")

st.sidebar.write("Model Prediction")

st.sidebar.write("Display Result")

# Footer

st.markdown("---")

st.caption("AI Fake News Detection System | NLP Internship Project")
