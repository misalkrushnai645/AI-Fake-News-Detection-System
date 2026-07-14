# Import Required Libraries

import streamlit as st
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

# Load NLP Resources

nltk.download("stopwords")
nltk.download("wordnet")

# Load Model and TF-IDF Vectorizer

model = joblib.load("model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize NLP Components

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

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

# Application Title

st.title("AI Fake News Detection System")

st.write(
    "This application predicts whether a news article is Fake News or Real News using Natural Language Processing and Machine Learning."
)

# Sidebar Information

st.sidebar.title("Project Information")

st.sidebar.write("Project: AI Fake News Detection System")
st.sidebar.write("Technology: NLP and Machine Learning")
st.sidebar.write("Algorithm: Linear Support Vector Classifier")
st.sidebar.write("Feature Extraction: TF-IDF Vectorization")
st.sidebar.write("Framework: Streamlit")

# News Analysis Section

st.subheader("News Analysis")

# Sample News Selection

sample_news = {

    "Custom News": ("", ""),

    "Real News Sample 1 - ISRO Chandrayaan-3": (
        "ISRO Successfully Lands Chandrayaan-3 Near Moon's South Pole",
        "ISRO successfully landed the Chandrayaan-3 mission near the Moon's south pole, making India the first country to achieve this milestone. The mission aims to study the lunar surface and conduct scientific experiments."
    ),

    "Real News Sample 2 - WHO": (
        "WHO Ends COVID-19 Global Health Emergency",
        "The World Health Organization announced that COVID-19 is no longer classified as a global public health emergency after reviewing global health data and vaccination progress."
    ),

    "Real News Sample 3 - NASA": (
        "NASA Launches Europa Clipper Mission",
        "NASA launched the Europa Clipper spacecraft to study Jupiter's moon Europa and investigate whether it has conditions that could support life beneath its icy surface."
    ),

    "Real News Sample 4 - RBI": (
        "Reserve Bank of India Keeps Repo Rate Unchanged",
        "The Reserve Bank of India decided to keep the repo rate unchanged after evaluating inflation trends, economic growth, and financial stability."
    ),

    "Real News Sample 5 - Election Commission": (
        "Election Commission Announces Schedule for State Elections",
        "The Election Commission released the official schedule for upcoming state assembly elections and confirmed that polling will take place in multiple phases."
    ),

    "Fake News Sample 1 - ₹10 Lakh": (
        "Government Announces ₹10 Lakh for Every Citizen",
        "A viral social media message claims that every Indian citizen will receive ₹10 lakh directly into their bank account without submitting any application."
    ),

    "Fake News Sample 2 - ₹500 Notes": (
        "RBI Declares All ₹500 Notes Invalid Overnight",
        "Posts circulating on social media falsely claim that all ₹500 currency notes will become invalid from midnight without any official announcement."
    ),

    "Fake News Sample 3 - NASA Darkness": (
        "NASA Confirms Three Days of Complete Darkness on Earth",
        "A viral online article falsely claims that Earth will experience three days of total darkness due to a rare astronomical event confirmed by NASA."
    ),

    "Fake News Sample 4 - WHO Coffee": (
        "WHO Announces Coffee Completely Cures Cancer",
        "Social media users are sharing false claims that the World Health Organization officially confirmed coffee can cure every type of cancer."
    ),

    "Fake News Sample 5 - ISRO Moon City": (
        "ISRO Discovers a City on the Moon",
        "A viral post falsely claims that ISRO found an ancient human city on the Moon during the Chandrayaan mission."
    )

}

sample_option = st.selectbox(
    "Choose Sample News",
    list(sample_news.keys())
)

sample_title, sample_content = sample_news[sample_option]

# User Input Section

news_title = st.text_input(
    "Enter News Title",
    value=sample_title
)

news_content = st.text_area(
    "Enter News Content",
    value=sample_content,
    height=250
)

# Analyze Button

analyze_button = st.button(
    "Analyze News"
)

# Prediction Section

if analyze_button:

    if news_title.strip() == "" and news_content.strip() == "":

        st.warning(
            "Please enter news title or news content."
        )

    else:

        complete_news = (
            news_title + " " + news_content
        )

        clean_text = preprocess_text(
            complete_news
        )

        vector = vectorizer.transform(
            [clean_text]
        )

        prediction = model.predict(
            vector
        )

        # NLP Processing Steps

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

        # Prediction Result

        st.subheader("Prediction Result")

        if prediction[0] == 0:

            st.error(
                "Prediction: Fake News"
            )

        else:

            st.success(
                "Prediction: Real News"
            )

# Footer Section

st.markdown("---")

st.caption(
    "AI Fake News Detection System | NLP Internship Project"
)
