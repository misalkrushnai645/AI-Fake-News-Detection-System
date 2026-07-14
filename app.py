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

sample_option = st.selectbox(
    "Choose Sample News",
    [
        "Custom News",
        "Real News Sample",
        "Fake News Sample"
    ]
)


sample_title = ""
sample_content = ""


if sample_option == "Real News Sample":

    sample_title = "Reuters reports US inflation slows"

    sample_content = (
        "Reuters reported that inflation in the United States "
        "slowed during the latest quarter according to official "
        "government statistics."
    )


elif sample_option == "Fake News Sample":

    sample_title = "Secret government plan exposed"

    sample_content = (
        "A viral social media post claims that the government "
        "has secretly banned all cash transactions from tomorrow."
    )



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
