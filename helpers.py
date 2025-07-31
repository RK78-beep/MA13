import pandas as pd
import numpy as np
import PyPDF2
import io

def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
        return pd.read_excel(file)
    elif file.name.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        data = {'Revenue': [1000], 'EBITDA': [200], 'Net Income': [100]}  # dummy
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format")

def process_files(uploaded_files):
    df1 = read_file(uploaded_files[0])
    df2 = read_file(uploaded_files[1])
    df1, df2 = df1.fillna(0), df2.fillna(0)
    return df1, df2

def predict_success(df1, df2, model):
    try:
        features = pd.concat([df1.mean(numeric_only=True), df2.mean(numeric_only=True)]).values.reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        prediction = int(prob >= 0.5)
        return prediction, prob
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

def generate_commentary(df1, df2, prediction, prob):
    revenue_a = df1['Revenue'].mean() if 'Revenue' in df1.columns else 'N/A'
    revenue_b = df2['Revenue'].mean() if 'Revenue' in df2.columns else 'N/A'
    result = "Proceed with deal 💼" if prediction == 1 else "Reconsider the deal ❌"
    commentary = f'''
    The predicted success probability for this M&A deal is **{prob:.2f}**.

    **Financial Highlights:**
    - Company A average revenue: {revenue_a}
    - Company B average revenue: {revenue_b}

    **System Verdict:** {result}
    '''
    return commentary
