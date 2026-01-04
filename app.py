import streamlit as st
import joblib
import numpy as np
from preprocess import clean_text
from features import extract_features
from scipy.sparse import hstack

# Load models
tfidf = joblib.load("models/tfidf.pkl")
clf_model = joblib.load("models/clf_voting.pkl")
reg_model = joblib.load("models/reg_voting.pkl")
le = joblib.load("models/label_encoder.pkl")

st.set_page_config(page_title="Problem Difficulty Predictor", layout="centered")

st.title("🧠 Programming Problem Difficulty Predictor")

st.markdown("Enter problem details below:")

title = st.text_input("Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")
sample_io = st.text_area("Sample Input/Output (optional)")

if st.button("Predict Difficulty"):
    full_text = " ".join([title, description, input_desc, output_desc, sample_io])
    full_text = clean_text(full_text)

    # TF-IDF
    text_vec = tfidf.transform([full_text])

    # Engineered features
    feat_df = extract_features(full_text)
    feat_vec = feat_df.values

    # Combine
    X_final = hstack([text_vec, feat_vec])

    # Predictions
    class_pred_enc = clf_model.predict(X_final)[0]
    class_pred = le.inverse_transform([class_pred_enc])[0]

    score_pred = reg_model.predict(X_final)[0]
    score_pred = np.clip(score_pred, 0, 10)

    st.success("Prediction Complete!")

    st.subheader("📊 Results")
    st.write(f"**Difficulty Class:** {class_pred}")
    st.write(f"**Difficulty Score:** {score_pred:.2f} / 10")
