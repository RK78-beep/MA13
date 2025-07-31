
import pandas as pd
import pdfplumber
import io

def parse_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        lines = text.split("\n")
        data = [line.split() for line in lines if line.strip()]
        df = pd.DataFrame(data)
        return df
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

def clean_and_align_data(df, model):
    expected = list(model.feature_names_in_)
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    return df[expected]

def generate_recommendations(df, prediction, prob):
    if prediction == 1:
        return f"This M&A deal appears promising with a success probability of {round(prob*100, 2)}%. Consider moving ahead, focusing on synergies and integration planning."
    else:
        return f"⚠️ The deal shows low probability of success ({round(prob*100, 2)}%). Re-evaluate the strategic fit, financial leverage, and cultural alignment between the firms."
