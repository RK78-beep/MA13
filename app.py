import streamlit as st
from helpers import (
    parse_file, preprocess_data, make_prediction, generate_commentary,
    plot_financials, explain_with_shap, calculate_synergy,
    score_esg_pmi, fetch_financial_news, convert_voice_to_text
)

st.set_page_config(page_title="Fusion IQ – M&A Deal Verdict AI", layout="wide")

st.title("🤖 Fusion IQ – M&A Deal Verdict AI")
st.markdown("Upload financials of two companies to predict M&A success, synergy, risk, and generate insights.")

# --- Login Simulation (Simple) ---
st.sidebar.title("🔐 Login")
user = st.sidebar.text_input("Username")
pwd = st.sidebar.text_input("Password", type="password")
if user != "admin" or pwd != "admin":
    st.warning("Enter valid credentials (admin/admin)")
    st.stop()

# --- Upload Section ---
st.header("📂 Upload Financials")
uploaded_file1 = st.file_uploader("Upload Company A File (CSV, Excel, or PDF)", type=["csv", "xlsx", "xls", "pdf"])
uploaded_file2 = st.file_uploader("Upload Company B File (CSV, Excel, or PDF)", type=["csv", "xlsx", "xls", "pdf"])

region = st.selectbox("🌍 Region (Optional)", ["", "North America", "Europe", "Asia", "South America", "Other"])
sector = st.text_input("🏢 Sector (Optional)")
environment = st.text_input("🌱 Deal Environment (Optional)")

if uploaded_file1 and uploaded_file2:
    with st.spinner("Processing..."):
        try:
            df1 = parse_file(uploaded_file1)
            df2 = parse_file(uploaded_file2)

            X, df_processed = preprocess_data(df1, df2, region, sector, environment)
            prediction, prob = make_prediction(X)

            st.subheader("🔎 Verdict:")
            st.metric("Success Probability", f"{prob*100:.2f}%")
            st.success("✅ Likely to Succeed" if prediction else "❌ Likely to Fail")

            st.subheader("💬 AI Commentary")
            st.markdown(generate_commentary(df_processed, prediction, prob))

            st.subheader("📊 Financial Comparison")
            plot_financials(df_processed)

            st.subheader("🧠 SHAP Explainability")
            shap_plot = explain_with_shap(X)
            st.pyplot(shap_plot)

            st.subheader("🔗 Synergy Score")
            synergy, synergy_comment = calculate_synergy(df1, df2)
            st.metric("Synergy Score", f"{synergy}/100")
            st.info(synergy_comment)

            esg_score, pmi_score = score_esg_pmi(df_processed)
            st.subheader("♻️ ESG Score & 🔄 PMI Risk")
            st.metric("ESG Compatibility", f"{esg_score}%")
            st.metric("PMI Risk", f"{pmi_score}%")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --- GPT-style Prompt ---
st.header("💡 Ask Anything")
prompt = st.text_input("Ask Fusion IQ a question about your deal, risks, valuation, or strategy:")
if prompt:
    st.markdown(generate_commentary(text_input=prompt))

# --- Voice-to-Text Input ---
st.subheader("🎙️ Or Speak Your Prompt")
if st.button("Start Listening"):
    text = convert_voice_to_text()
    if text:
        st.success(f"You said: {text}")
        st.markdown(generate_commentary(text_input=text))
    else:
        st.warning("Could not recognize speech. Try again.")

# --- Real-time News Section ---
st.header("📰 Latest M&A News")
news_items = fetch_financial_news("merger acquisition")
for article in news_items:
    st.markdown(f"- [{article['title']}]({article['url']})")

# --- Footer ---
st.markdown("---")
st.caption("Built with 💼 for smarter M&A decisions. Fusion IQ © 2025")
