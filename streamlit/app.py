import streamlit as st
import pickle
import re

# Load model & vectorizer
model = pickle.load(open('streamlit/model.pkl', 'rb'))
vectorizer = pickle.load(open('streamlit/vectorizer.pkl', 'rb'))

# Fungsi preprocessing (HARUS sama dengan training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# UI
st.set_page_config(page_title="AI Text Detection", layout="centered")

st.title("🤖 AI vs Human Text Detection")
st.write("Masukkan teks untuk mendeteksi apakah dibuat oleh AI atau manusia.")

text = st.text_area("Input Teks:", height=200)

if st.button("Deteksi"):
    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        if prediction == 1:
            st.error("⚠️ AI Generated")
            st.write(f"Confidence: {max(prob)*100:.2f}%")
        else:
            st.success("✅ Human Written")
            st.write(f"Confidence: {max(prob)*100:.2f}%")
