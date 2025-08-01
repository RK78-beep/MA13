import pandas as pd
import numpy as np
import base64
import PyPDF2
import io
import shap
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import requests
import json
from datetime import datetime
import tempfile

# --- File Parsing Functions ---

def parse_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    else:
        return pd.DataFrame()

def extract_text_from_pdf(uploaded_pdf):
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return pd.DataFrame({'ExtractedText': [text]})

# --- Feature Engineering & Mapping ---

expected_features = [
    'Revenue', 'EBITDA', 'Net Income', 'Total Assets', 'Total Liabilities',
    'Equity', 'Cash Flow', 'CapEx', 'Region', 'Sector', 'Deal Size'
]

def clean_and_map_features(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    feature_map = {
        'revenue': 'Revenue',
        'ebitda': 'EBITDA',
        'net income': 'Net Income',
        'assets': 'Total Assets',
        'liabilities': 'Total Liabilities',
        'equity': 'Equity',
        'cash': 'Cash Flow',
        'capex': 'CapEx',
        'region': 'Region',
        'sector': 'Sector',
        'deal': 'Deal Size'
    }

    mapped_df = pd.DataFrame()
    for key, value in feature_map.items():
        for col in df.columns:
            if key in col and value not in mapped_df.columns:
                mapped_df[value] = df[col]
                break

    return mapped_df

# --- DCF Calculator ---

def calculate_dcf(revenue, growth_rate=0.05, discount_rate=0.1, years=5):
    future_cash_flows = [revenue * (1 + growth_rate) ** i for i in range(1, years + 1)]
    discounted_cash_flows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(future_cash_flows, 1)]
    terminal_value = future_cash_flows[-1] * (1 + growth_rate) / (discount_rate - growth_rate)
    terminal_discounted = terminal_value / (1 + discount_rate) ** years
    return sum(discounted_cash_flows) + terminal_discounted

# --- GPT-style Commentary ---

def generate_commentary(input_data):
    if 'Revenue' in input_data and 'Net Income' in input_data:
        rev = input_data['Revenue']
        ni = input_data['Net Income']
        margin = round(ni / rev * 100, 2) if rev else 0
        return f"Company is operating at a net margin of {margin}%, suggesting {'strong' if margin > 15 else 'moderate'} profitability."
    return "Insufficient data for commentary."

# --- SHAP Explainability Plot ---

def generate_shap_plot(model, X, explainer=None):
    if explainer is None:
        explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    fig = shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name, bbox_inches='tight')
    plt.close()
    return tmpfile.name

# --- Synergy Estimator ---

def estimate_synergy(df1, df2):
    if 'Revenue' in df1 and 'Revenue' in df2:
        synergy = 0.05 * (df1['Revenue'] + df2['Revenue'])
        return synergy
    return None

# --- Real-Time News ---

def get_financial_news(api_key, query='merger acquisition', page_size=5):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [
            {
                'title': article['title'],
                'url': article['url'],
                'source': article['source']['name'],
                'publishedAt': article['publishedAt']
            } for article in articles
        ]
    else:
        return [{'title': 'News fetch failed', 'url': '', 'source': '', 'publishedAt': ''}]
