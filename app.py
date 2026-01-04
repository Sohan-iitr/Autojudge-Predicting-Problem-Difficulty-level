import streamlit as st
import joblib
from scipy.sparse import hstack
import numpy as np

from preprocess import preprocess_text
from features import extract_features

st.set_page_config(
    page_title="Problem Difficulty Predictor",
    layout="centered"
)

@st.cache_resource
def load_models():
    clf = joblib.load("models/clf_voting.pkl")
    reg = joblib.load("models/reg_voting.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return clf, reg, tfidf, le

clf_model, reg_model, tfidf, le = load_models()

st.title("🧠 Programming Problem Difficulty Predictor")
st.markdown("Enter problem details below:")

title = st.text_input("Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")
sample_io = st.text_area("Sample Input/Output (optional)")

if st.button("Predict Difficulty"):
    try:
        user_text = " ".join([
            title or "",
            description or "",
            input_desc or "",
            output_desc or "",
            sample_io or ""
        ])

        clean_text = preprocess_text(user_text)

        X_text = tfidf.transform([clean_text])
        X_feat = extract_features(clean_text).astype(np.float32)

        X_final = hstack([X_text, X_feat.values])

        class_pred_enc = clf_model.predict(X_final)[0]
        class_pred = le.inverse_transform([class_pred_enc])[0]

        score_pred = reg_model.predict(X_final)[0]
        score_pred = float(np.clip(score_pred, 0, 10))


        st.success("Prediction Complete!")
        st.write(f"### Difficulty: {class_pred}")
        st.write(f"### Score: {score_pred:.2f}")

    except Exception as e:
        st.error("Something went wrong during prediction.")
        st.exception(e)
