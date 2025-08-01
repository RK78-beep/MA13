import streamlit as st
import pandas as pd
import os
import tempfile
from helpers import (
    load_model, process_uploaded_file, predict_success,
    generate_commentary, show_shap_explanation,
    synergy_analysis, esg_check, pmi_risk_score,
    extract_text_from_voice, fetch_latest_financial_news
)
from datetime import datetime

# Page config
st.set_page_config(
    page_title="M&A Deal Verdict AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 M&A Deal Verdict AI")
st.markdown("Upload financials of two companies or speak a prompt. Get deal success prediction, synergy, ESG, SHAP & more.")

# Sidebar
st.sidebar.header("Upload Company Files")
file1 = st.sidebar.file_uploader("📁 Upload Company A (CSV, Excel, PDF)", type=['csv', 'xlsx', 'xls', 'pdf'], key="file1")
file2 = st.sidebar.file_uploader("📁 Upload Company B (CSV, Excel, PDF)", type=['csv', 'xlsx', 'xls', 'pdf'], key="file2")
voice_prompt = st.sidebar.text_area("🎤 Or describe the deal (voice-to-text)", "")
if st.sidebar.button("🗣️ Transcribe Voice"):
    transcribed = extract_text_from_voice(voice_prompt)
    st.sidebar.success(f"Transcribed: {transcribed}")

# Load model
model = load_model("model.pkl")

# File processing
if file1 and file2:
    st.subheader("1️⃣ Processed Financials")
    try:
        df1 = process_uploaded_file(file1)
        df2 = process_uploaded_file(file2)
        st.success("Files uploaded and processed successfully.")
        st.dataframe(df1.head(3))
        st.dataframe(df2.head(3))
    except Exception as e:
        st.error(f"File processing failed: {e}")
        st.stop()

    st.subheader("2️⃣ Deal Success Prediction")
    try:
        success_prob, X_merged = predict_success(df1, df2, model)
        st.metric(label="Predicted Deal Success Rate", value=f"{success_prob*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("3️⃣ GPT-style Deal Summary")
    st.markdown(generate_commentary(df1, df2, success_prob))

    st.subheader("4️⃣ Synergy & Strategic Fit")
    synergy_score = synergy_analysis(df1, df2)
    st.metric("Estimated Synergy Score", synergy_score)

    st.subheader("5️⃣ ESG & PMI Risk Analysis")
    esg = esg_check(df1, df2)
    pmi = pmi_risk_score(df1, df2)
    st.write(f"♻️ ESG Compatibility: {esg}")
    st.write(f"🔁 Post-Merger Integration Risk Score: {pmi}")

    st.subheader("6️⃣ SHAP Explainability")
    show_shap_explanation(model, X_merged)

    st.success("✅ Analysis Complete")

# Optional News Feed
st.subheader("📰 Latest Financial News")
news = fetch_latest_financial_news(api_key="f18e256bcfba46758e59667478fcf462")
for article in news:
    st.markdown(f"- [{article['title']}]({article['url']})")

# Footer
st.markdown("---")
st.markdown("© 2025 M&A Deal Verdict AI | Built with ❤️ using Streamlit")
