# Fake News Detection System

## Project Overview

Fake News Detection System is a Natural Language Processing (NLP) based Machine Learning project that classifies news articles as Fake News or Real News.

The project uses text preprocessing, TF-IDF Vectorization and Linear Support Vector Classifier (LinearSVC) to perform binary text classification.

---

## Objectives

- Detect fake and real news automatically.
- Perform text preprocessing using NLP techniques.
- Convert text into numerical features using TF-IDF Vectorization.
- Train a Machine Learning model.
- Develop a user-friendly Streamlit web application.

---

## Features

- Fake and Real News Prediction
- Text Preprocessing
- TF-IDF Vectorization
- LinearSVC Classification Model
- Streamlit Web Application
- User Friendly Interface

---

## Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

NLP Techniques

- Text Cleaning
- Tokenization
- Lemmatization
- TF-IDF Vectorization

Machine Learning Algorithm

- Linear Support Vector Classifier (LinearSVC)

---

## Project Workflow

Dataset

↓

Data Preprocessing

↓

Text Cleaning

↓

Tokenization

↓

Lemmatization

↓

TF-IDF Vectorization

↓

Model Training

↓

Model Evaluation

↓

Prediction

↓

Streamlit Application

---

## Project Structure

```
Fake-News-Detection-System

│── app.py
│── README.md
│── requirements.txt

│── model
│ ├── model.pkl
│ └── tfidf_vectorizer.pkl

│── notebooks
│ ├── 01_Data_Preprocessing.ipynb
│ └── 03_Model_Training_Evaluation.ipynb
```

---

## How to Run

Install Required Libraries

```bash
pip install -r requirements.txt
```

Run Streamlit Application

```bash
streamlit run app.py
```

---

## Model Details

Feature Extraction

- TF-IDF Vectorization

Classification Algorithm

- Linear Support Vector Classifier (LinearSVC)

Problem Type

- Binary Text Classification

---

## Output

The application predicts whether the given news article is:

- Fake News
- Real News

---

## Future Scope

- Improve prediction accuracy
- Support multilingual news
- Deploy on cloud platform
- Add deep learning models