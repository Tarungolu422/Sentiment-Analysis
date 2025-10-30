# =========================
# Streamlit App: Restaurant Review Sentiment Analysis
# Author: Tarun Kumar Rathore
# Features: Multiple ML models, TF-IDF, NLTK preprocessing,
# Prediction, Probability, Metrics, ROC curve
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# =========================
# NLTK Downloads
# =========================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =========================
# App Layout & Styling
# =========================
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

page_bg = """
<style>
body {
background: linear-gradient(to right, #FFDEE9, #B5FFFC);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    height: 3em;
    width: 100%;
    border-radius: 10px;
}
.stTextInput>div>input {
    height: 3em;
    font-size: 16px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# Sidebar Info
# =========================
with st.sidebar:
    st.header("📌 Instructions")
    st.write("""
    1. Select a trained model.
    2. Enter a restaurant review.
    3. Click 'Predict Sentiment' to see the result.
    """)
    st.markdown("---")
    st.header("ℹ️ About")
    st.write(
        "Interactive sentiment analysis of restaurant reviews using multiple ML models.")
    st.markdown("---")
    st.header("👤 Developer")
    st.write("Tarun Kumar Rathore")

# =========================
# Load TF-IDF Vectorizer (Hidden Status)
# =========================
TFIDF_PATH = os.path.join("models", "tfidf.pkl")

try:
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error(
        "❌ TF-IDF vectorizer file not found. Make sure 'tfidf.pkl' exists in the 'models' folder.")
    st.stop()

# =========================
# Folder Paths
# =========================
MODEL_FOLDER = "models"
METRICS_FOLDER = "metrics"
ROC_FOLDER = "roc_curves"

# =========================
# Load Available Files
# =========================
model_files = [f for f in os.listdir(
    MODEL_FOLDER) if f.endswith(".pkl") and f != "tfidf.pkl"]

if not model_files:
    st.error("❌ No model files found in 'models/' folder.")
    st.stop()

# Remove .pkl extension for dropdown display
model_names = [os.path.splitext(f)[0] for f in model_files]

# =========================
# Main Title & Model Selection
# =========================
st.title("🍽️ Restaurant Review Sentiment Analysis")
st.subheader("Predict restaurant review sentiment using various ML models")

# 🧠 Model dropdown appears AFTER the title
selected_model_name = st.selectbox("🎯 Select Model", model_names)
selected_model_file = f"{selected_model_name}.pkl"

# =========================
# Load Selected Model
# =========================
with open(os.path.join(MODEL_FOLDER, selected_model_file), "rb") as f:
    selected_model = pickle.load(f)

# =========================
# Text Input
# =========================
st.subheader(f"Analyze sentiment using **{selected_model_name}** model")
review_text = st.text_area("Enter a Restaurant Review:", height=150)

# =========================
# NLTK Preprocessing
# =========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


# =========================
# Prediction
# =========================
if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            processed_text = preprocess_text(review_text)
            vectorized = tfidf.transform([processed_text]).toarray()
            pred = selected_model.predict(vectorized)[0]

            try:
                prob = selected_model.predict_proba(vectorized)[0][1]
                prob_percent = round(prob * 100, 2)
            except AttributeError:
                prob_percent = None

            if pred == 1:
                st.success(
                    f"Positive Review 👍 ({prob_percent}% confidence)" if prob_percent else "Positive Review 👍")
                st.balloons()
            else:
                st.error(
                    f"Negative Review 👎 ({100 - prob_percent}% confidence)" if prob_percent else "Negative Review 👎")
                st.snow()

# =========================
# Metrics Expander (Auto Link)
# =========================
with st.expander("📊 Show Model Evaluation Metrics"):
    metrics_path = os.path.join(
        METRICS_FOLDER, f"{selected_model_name}_metrics.pkl")
    if os.path.exists(metrics_path):
        metrics = pickle.load(open(metrics_path, "rb"))
        metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Value"})
        st.table(metrics_df)
    else:
        st.info("Metrics file not found for this model.")

# =========================
# ROC Expander (Auto Link)
# =========================
with st.expander("📈 Show ROC Curve"):
    roc_path = os.path.join(ROC_FOLDER, f"{selected_model_name}_roc.pkl")
    if os.path.exists(roc_path):
        fpr, tpr = pickle.load(open(roc_path, "rb"))
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{selected_model_name}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve - {selected_model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(plt)
    else:
        st.info("ROC curve file not found for this model.")
