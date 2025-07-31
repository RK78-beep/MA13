import streamlit as st
from helpers import process_file, generate_gpt_commentary, plot_shap_summary, predict_success
import joblib

st.set_page_config(page_title="M&A Deal Verdict AI", layout="wide")

st.title("🤖 M&A Deal Verdict AI")
st.markdown("Upload financials of two companies (CSV, Excel, or PDF) to assess M&A deal success.")

uploaded_files = st.file_uploader("Upload files", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 2:
        st.warning("Please upload exactly two files.")
    else:
        st.success("Files uploaded successfully.")
        try:
            df1 = process_file(uploaded_files[0])
            df2 = process_file(uploaded_files[1])
            st.subheader("Parsed Financials")
            st.write("Company 1")
            st.dataframe(df1)
            st.write("Company 2")
            st.dataframe(df2)

            model = joblib.load("model.pkl")
            prediction, probability = predict_success(df1, df2, model)
            st.subheader("Prediction Result")
            st.write(f"Success Probability: {probability:.2%}")
            st.write("Deal Verdict:", "✅ Recommended" if prediction == 1 else "❌ Not Recommended")

            st.subheader("GPT-style Deal Analysis")
            st.markdown(generate_gpt_commentary(df1, df2, prediction, probability), unsafe_allow_html=True)

            st.subheader("Explainability (SHAP)")
            plot_shap_summary(df1, df2, model)
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
