# ============================================================
# AI Fake News Detection System
# NLP + Machine Learning + Streamlit
# ============================================================

# Import Libraries

import streamlit as st
import joblib
import re
import string
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="AI Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# ============================================================
# Load Model and NLP Resources
# ============================================================
@st.cache_resource
def load_resources():

    model = joblib.load(
        "model/model_confidence.pkl"
    )

    vectorizer = joblib.load(
        "model/tfidf_vectorizer.pkl"
    )

    nlp = spacy.load(
        "en_core_web_sm"
    )

    return model, vectorizer, nlp

model, vectorizer, nlp = load_resources()

# ============================================================
# Download NLTK Resources
# ============================================================
nltk.download(
    "punkt",
    quiet=True
)

nltk.download(
    "punkt_tab",
    quiet=True
)

nltk.download(
    "stopwords",
    quiet=True
)

stop_words = set(
    stopwords.words("english")
)

# ============================================================
# NLP Preprocessing Function
# ============================================================
def preprocess_text(text):

    # Convert to lowercase

    text = text.lower()

    # Remove URLs

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove HTML tags

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove punctuation

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers

    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Tokenization

    tokens = word_tokenize(
        text
    )

    # Stopword Removal

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization using spaCy

    doc = nlp(
        " ".join(tokens)
    )

    clean_tokens = [
        token.lemma_
        for token in doc
    ]

    clean_text = " ".join(
        clean_tokens
    )

return clean_text

# ============================================================
# Prediction and Result Section
# ============================================================
if analyze_button:

    if news_content.strip() == "":

     st.warning(
        "⚠️ Please enter news content first."
     )

else:
     
    with st.spinner(
        "🔍 Analyzing News using NLP..."
     ):


     # Combine title and content

        complete_news = (
             news_title +
             " " +
             news_content
     )

     # NLP Processing

     clean_text = preprocess_text(
            complete_news
     )

     # Show NLP Steps

     st.subheader(
            "🧠 NLP Processing Steps"
     )

     steps = [
            "✅ Convert text to lowercase",
            "✅ Remove URLs and special characters",
            "✅ Tokenization",
            "✅ Stopword Removal",
            "✅ Lemmatization",
            "✅ TF-IDF Feature Extraction"
     ]
for step in steps:

     st.write(step)

     # TF-IDF Transformation

     vector = vectorizer.transform(
            [clean_text]
     )

     # Prediction

     prediction = model.predict(
            vector
     )

     # Confidence

     probability = model.predict_proba(
            vector
     )

     confidence = max(
        probability[0]
     ) * 100

     st.divider()

# ====================================================
# Result Card
# ====================================================
if prediction[0] == 0:

     st.error(
            "❌ FAKE NEWS DETECTED"
     )

        result = "Fake News"

        message = "Low credibility news detected."

else:

     st.success(
            "✅ REAL NEWS DETECTED"
     )

        result = "Real News"

        message = "News appears to be reliable."

    st.subheader(
        "📊 News Credibility Meter"
    )

    st.progress(
        int(confidence)
    )

    st.metric(
         "Confidence Score",
         f"{confidence:.2f}%"
    )

    st.info(
        message
    )

# ====================================================
# Processed Text Preview
# ====================================================
    st.subheader(
        "📝 NLP Processed Text Preview"
    )


    st.write(
        clean_text[:500]
    )

# ====================================================
# Save History
# ====================================================
    st.session_state.history.append(
        {
             "Prediction": result,
             "Confidence": f"{confidence:.2f}%"
        }
    )

# ============================================================
# Prediction History
# ============================================================
if len(st.session_state.history) > 0:

    st.divider()

    st.subheader(
        "📋 Prediction History"
    )

    st.table(
        st.session_state.history
    )

# ============================================================
# Footer
# ============================================================
st.divider()

st.caption(
    "AI Fake News Detection System | NLP + Machine Learning Project"
)

# ============================================================
# Sidebar UI
# ============================================================
with st.sidebar:

    st.title("📰 AI Fake News Detector")

    st.write(
        """
        NLP Based Fake News
        Classification System
        """
    )
    st.divider()

    st.subheader("Technology Used")

    st.write(
        """
        ✅ Natural Language Processing  
        ✅ TF-IDF Vectorization  
        ✅ Machine Learning  
        ✅ Streamlit  
        """
    )
    st.divider()

    st.info(
        "Enter news content and click Analyze button."
    )
    
# ============================================================
# Main Header
# ============================================================

st.markdown(
    """
    <h1 style='text-align:center;'>
    📰 AI Fake News Detection System
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align:center; font-size:18px;'>
    Detect Fake and Real News using NLP and Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

# ============================================================
# Prediction History Storage
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []

# ============================================================
# News Input Section
# ============================================================

news_title = st.text_input(
    "📝 Enter News Title"
)


news_content = st.text_area(
    "📰 Enter News Content",
    height=250,
    placeholder="Paste news article here..."
)

# ============================================================
# Buttons
# ============================================================

col1, col2 = st.columns(2)


with col1:

    analyze_button = st.button(
        "🔍 Analyze News",
        use_container_width=True
    )


with col2:

    clear_button = st.button(
        "🧹 Clear Input",
        use_container_width=True
    )

if clear_button:

    st.session_state.clear()

    st.rerun()

# ============================================================
# Sample News
# ============================================================

with st.expander(
    "📌 Try Sample News"
):

    st.write(
        "The government announced a new education policy to improve schools."
    )
