import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from features import get_manual_features


@st.cache_resource
def load_models():
    clf = joblib.load("classifier.pkl")
    reg = joblib.load("regressor.pkl")
    tfidf = joblib.load("tfidf.pkl")
    scaler = joblib.load("scaler.pkl") 
    return clf, reg, tfidf, scaler

clf, reg, tfidf, scaler = load_models()

st.title(" AutoJudge – Problem Difficulty Predictor")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

if st.button("Predict"):
    if not (desc.strip() or inp.strip() or out.strip()):
        st.warning("Please enter some text.")
    else:
        combined = desc + " " + inp + " " + out

        X_text = tfidf.transform([combined])
        
        manual_features_list = get_manual_features(combined)
        X_manual_raw = np.array([manual_features_list])
        
        X_manual_scaled = scaler.transform(X_manual_raw)

        X = hstack([X_text, X_manual_scaled])

        predicted_class = clf.predict(X)[0]
        predicted_score = reg.predict(X)[0]

        st.success(f"Predicted Difficulty Class: {predicted_class}")
        st.success(f"Predicted Difficulty Score: {predicted_score:.2f}")