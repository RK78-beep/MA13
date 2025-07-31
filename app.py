import streamlit as st
import pandas as pd
import os
from helpers import process_file, run_model, generate_commentary, show_shap, generate_downloadable_report

st.set_page_config(page_title="M&A Deal Verdict AI", layout="wide")

st.title("🤝 M&A Deal Verdict AI")
st.write("Upload financials of two companies (CSV, Excel, or PDF) to analyze the M&A deal potential.")

uploaded_files = st.file_uploader("Upload two company files (CSV, XLSX, or PDF)", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    try:
        df1 = process_file(uploaded_files[0])
        df2 = process_file(uploaded_files[1])

        if df1 is not None and df2 is not None:
            st.success("Files successfully processed.")

            combined_df = pd.concat([df1, df2], axis=1)
            st.subheader("📊 Combined Financial Overview")
            st.dataframe(combined_df)

            prediction, probability = run_model(combined_df)
            st.subheader("📈 Prediction Outcome")
            st.write(f"**Success Probability:** {round(probability * 100, 2)}%")
            verdict = "✅ Likely to Succeed" if prediction == 1 else "❌ Unlikely to Succeed"
            st.markdown(f"### Deal Verdict: {verdict}")

            st.subheader("💬 GPT-style Commentary")
            st.write(generate_commentary(combined_df, prediction, probability))

            st.subheader("🧠 SHAP Explainability")
            shap_fig = show_shap(combined_df)
            st.pyplot(shap_fig)

            st.subheader("📥 Download Full Report")
            report = generate_downloadable_report(combined_df, prediction, probability)
            st.download_button("Download Report", report, file_name="M&A_Report.txt")
        else:
            st.error("One of the files could not be parsed properly.")
    except Exception as e:
        st.error(f"Error occurred: {e}")
else:
    st.info("Please upload exactly 2 files.")
