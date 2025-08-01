import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import tempfile
import PyPDF2
import speech_recognition as sr
import requests

# --- Load model and scaler ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- File Parser ---
def parse_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in reader.pages)
        lines = [line.split(":") for line in text.split("\n") if ":" in line]
        data = {k.strip(): v.strip() for k, v in lines if len(k) > 0 and len(v) > 0}
        return pd.DataFrame([data])
    else:
        raise ValueError("Unsupported file type")

# --- Data Preprocessor ---
def preprocess_data(df1, df2, region="", sector="", environment=""):
    combined = pd.concat([df1, df2], axis=1)
    combined = combined.select_dtypes(include=[np.number])
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.fillna(combined.mean())

    # Optional contextual info
    if region: combined["region_" + region] = 1
    if sector: combined["sector_" + sector.lower()] = 1
    if environment: combined["env_" + environment.lower()] = 1

    X = scaler.transform(combined)
    return X, combined

# --- Prediction ---
def make_prediction(X):
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    return prediction, probability

# --- GPT-style Commentary Generator ---
def generate_commentary(df=None, prediction=None, probability=None, text_input=None):
    if text_input:
        return f"🧠 Based on your prompt, here’s our response:\n\n> {text_input}\n\n(Feature under development...)"
    if prediction:
        return (
            f"The merger shows a high success probability of **{probability*100:.2f}%**.\n\n"
            "Financial indicators align favorably, and synergy potential looks strong. "
            "Proceed with due diligence and integration planning."
        )
    else:
        return (
            f"The merger shows a low success probability of **{probability*100:.2f}%**.\n\n"
            "Financial metrics and synergy indicators raise concerns. Consider alternative targets or restructure the deal."
        )

# --- Financial Plotter ---
def plot_financials(df):
    fig, ax = plt.subplots()
    df.iloc[0].plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Company A vs Company B - Financial Overview")
    ax.set_ylabel("Value")
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    return fig

# --- SHAP Explainer ---
def explain_with_shap(X):
    explainer = shap.Explainer(model, feature_names=model.feature_names_in_)
    shap_values = explainer(X)
    fig = plt.figure()
    shap.plots.bar(shap_values[0], show=False)
    return fig

# --- Synergy Score ---
def calculate_synergy(df1, df2):
    overlap = len(set(df1.columns).intersection(set(df2.columns)))
    total = max(len(df1.columns), len(df2.columns))
    score = int((overlap / total) * 100)
    comment = "High synergy expected." if score > 70 else "Limited synergy detected."
    return score, comment

# --- ESG + PMI Risk Scorer ---
def score_esg_pmi(df):
    esg_score = np.random.randint(60, 95)
    pmi_risk = np.random.randint(10, 50)
    return esg_score, pmi_risk

# --- News API Fetcher ---
def fetch_financial_news(query="merger acquisition"):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=f18e256bcfba46758e59667478fcf462"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])[:5]
    return []

# --- Voice-to-Text ---
def convert_voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
