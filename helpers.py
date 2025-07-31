import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import shap
import matplotlib.pyplot as plt
import streamlit as st
import joblib

def process_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        # Fake parser from PDF text
        data = {
            "Revenue": [1000],
            "EBITDA": [150],
            "Net Income": [80],
            "Total Assets": [3000],
            "Debt": [1200],
            "Equity": [1800],
            "Region": ["US"],
            "Sector": ["Tech"]
        }
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format.")
    return df

def predict_success(df1, df2, model):
    features = ["Revenue", "EBITDA", "Net Income", "Total Assets", "Debt", "Equity"]
    x1 = df1[features].mean()
    x2 = df2[features].mean()
    X = pd.DataFrame([abs(x1 - x2)])
    probability = model.predict_proba(X)[0][1]
    prediction = int(probability > 0.5)
    return prediction, probability

def generate_gpt_commentary(df1, df2, prediction, probability):
    sentiment = "positive" if prediction == 1 else "cautious"
    commentary = f"""
    ### GPT-style Commentary
    Based on the provided financials, this M&A deal has a **{probability:.2%} chance of success**.
    The model has assessed the financial synergy and operational alignment between both firms as **{sentiment}**.
    It is {"advisable to proceed" if prediction == 1 else "recommended to reassess the deal structure or search for better-fit targets"}.
    """
    return commentary

def plot_shap_summary(df1, df2, model):
    features = ["Revenue", "EBITDA", "Net Income", "Total Assets", "Debt", "Equity"]
    x1 = df1[features].mean()
    x2 = df2[features].mean()
    X = pd.DataFrame([abs(x1 - x2)])
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
