# AutoJudge: Predicting Programming Problem Difficulty

AutoJudge is a machine learning system that predicts the **difficulty class**
(Easy / Medium / Hard) and a **numerical difficulty score** for programming
problems using only their textual descriptions.

The project is inspired by online competitive programming platforms where
difficulty labeling is subjective and time-consuming.

---
demo video link:  
https://drive.google.com/drive/folders/1eLseWfE6EQKvZpkXYxRKQ80oaCzbHC2a?usp=sharing

models pkl file link (since large unable to upload to github):  
 https://drive.google.com/drive/folders/1CIGuZFMnDzFtFnBbUau-buLVP2uNHBQA?usp=sharing  

## 🚀 Features

- Predicts **difficulty class** (Easy / Medium / Hard)
- Predicts **difficulty score** (0–10)
- Uses only problem text (no metadata)
- Combines **NLP + feature engineering**
- Includes a **Streamlit web interface**

---

## 🧠 Approach

### 1. Data Preprocessing
- Combined all textual fields into a single input
- Lowercasing and basic text cleaning
- No heavy linguistic preprocessing to preserve semantics

### 2. Feature Extraction
- TF-IDF vectorization (top 10,000 terms)
- Engineered features:
  - Text length metrics
  - Numeric and symbol counts
  - Keyword-based indicators (dp, graph, greedy, etc.)

### 3. Models

#### Classification
- Logistic Regression
- Random Forest
- XGBoost
- **Soft Voting Ensemble**

#### Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- **Voting Regressor Ensemble**

### 4. Evaluation
- Classification: Accuracy, Macro F1, Confusion Matrix
- Regression: R², RMSE

---

## 🌐 Web Interface

A simple Streamlit UI allows users to paste a new problem description and get:
- Predicted difficulty class
- Predicted difficulty score

---

## 🛠️ How to Run

#### Install dependencies
```bash
pip install -r requirements.txt
```
Then download model files from the drive mentioned above (large hence unable to upload on github)

Run the Streamlit app
```bash
python -m streamlit run app.py
```
Author  
Sohan Awate