import streamlit as st
from helpers import process_files, predict_success, generate_commentary
import pandas as pd
import pickle

st.set_page_config(page_title="M&A Deal Verdict AI", layout="wide")
st.title("🤝 M&A Deal Verdict AI")

uploaded_files = st.file_uploader("Upload two company financials (CSV, Excel, or PDF)", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    try:
        df1, df2 = process_files(uploaded_files)
        st.success("Files processed successfully.")
        st.write("📄 **Company A Financials**")
        st.dataframe(df1)
        st.write("📄 **Company B Financials**")
        st.dataframe(df2)

        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        prediction, prob = predict_success(df1, df2, model)
        st.subheader("🔍 Deal Success Prediction")
        st.write(f"**Prediction:** {'✅ Likely to Succeed' if prediction==1 else '❌ Likely to Fail'}")
        st.write(f"**Probability of Success:** {prob:.2f}")

        commentary = generate_commentary(df1, df2, prediction, prob)
        st.subheader("💡 GPT-Style Deal Insights")
        st.markdown(commentary)

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("Please upload **two** company financial files.")
