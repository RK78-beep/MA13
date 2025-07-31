import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import io
from PyPDF2 import PdfReader

model = joblib.load("model.pkl")

def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            data = extract_financials_from_text(text)
            return pd.DataFrame([data])
    except Exception:
        return None

def extract_financials_from_text(text):
    return {
        "revenue": 1000,
        "ebitda": 200,
        "net_income": 100,
        "assets": 5000,
        "liabilities": 3000
    }

def run_model(df):
    df = df.select_dtypes(include=[np.number]).fillna(0)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability

def generate_commentary(df, prediction, probability):
    sentiment = "positive" if prediction == 1 else "negative"
    return f"Based on financials, this M&A deal has a {round(probability * 100)}% chance of success. The model sees this as a {sentiment} outlook."

def show_shap(df):
    explainer = shap.Explainer(model)
    shap_values = explainer(df)
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    return fig

def generate_downloadable_report(df, prediction, probability):
    output = io.StringIO()
    verdict = "Success" if prediction == 1 else "Failure"
    output.write("M&A Deal Verdict Report\n")
    output.write("========================\n")
    output.write(f"Verdict: {verdict}\n")
    output.write(f"Success Probability: {round(probability*100, 2)}%\n\n")
    output.write("Input Summary:\n")
    output.write(df.to_string())
    return output.getvalue()
