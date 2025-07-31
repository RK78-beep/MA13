
import streamlit as st
import pandas as pd
import joblib
from helpers import parse_uploaded_file, clean_and_align_data, generate_recommendations

st.set_page_config(page_title="M&A Deal Verdict AI", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🤖 M&A Deal Verdict AI")
st.write("Upload financial documents of two companies to evaluate M&A success probability and get intelligent recommendations.")

uploaded_file1 = st.file_uploader("Upload Company 1 Financials (PDF, Excel, or CSV)", type=["pdf", "csv", "xls", "xlsx"])
uploaded_file2 = st.file_uploader("Upload Company 2 Financials (PDF, Excel, or CSV)", type=["pdf", "csv", "xls", "xlsx"])

if uploaded_file1 and uploaded_file2:
    try:
        df1 = parse_uploaded_file(uploaded_file1)
        df2 = parse_uploaded_file(uploaded_file2)

        combined_df = pd.concat([df1, df2]).reset_index(drop=True)
        clean_df = clean_and_align_data(combined_df, model)

        prediction = model.predict(clean_df)[0]
        probability = model.predict_proba(clean_df)[0][1]

        st.subheader("📊 M&A Deal Verdict")
        st.write(f"**Prediction:** {'✅ Likely to Succeed' if prediction == 1 else '❌ Likely to Fail'}")
        st.write(f"**Probability of Success:** {round(probability * 100, 2)}%")

        st.subheader("🧠 GPT-Style Strategic Recommendations")
        st.markdown(generate_recommendations(clean_df, prediction, probability))

    except Exception as e:
        st.error(f"An error occurred while processing the files: {str(e)}")
