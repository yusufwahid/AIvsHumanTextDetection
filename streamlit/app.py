import streamlit as st
import pickle
import re
import ast
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.sparse import hstack
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ========================
# CACHE: Load model
# ========================
@st.cache_resource
def load_model():
    model = pickle.load(open(BASE_DIR / 'model.pkl', 'rb'))
    vectorizer = pickle.load(open(BASE_DIR / 'vectorizer.pkl', 'rb'))
    return model, vectorizer

# ========================
# CACHE: Load & proses dataset
# ========================
@st.cache_data
def load_dataset():
    try:
        df_raw = pd.read_csv(BASE_DIR / 'dataset_hc3.csv', engine='python', on_bad_lines='skip')

        def clean_list_string(text):
            try:
                val = ast.literal_eval(str(text))
                return val[0] if isinstance(val, list) and len(val) > 0 else str(text)
            except:
                return str(text)

        df_raw['human_answers_clean'] = df_raw['human_answers'].apply(clean_list_string)
        df_raw['chatgpt_answers_clean'] = df_raw['chatgpt_answers'].apply(clean_list_string)

        df_human = df_raw[['human_answers_clean']].rename(columns={'human_answers_clean': 'text'})
        df_human['label'] = 0
        df_chatgpt = df_raw[['chatgpt_answers_clean']].rename(columns={'chatgpt_answers_clean': 'text'})
        df_chatgpt['label'] = 1

        df = pd.concat([df_human, df_chatgpt], ignore_index=True)
        df = df[df['text'].str.strip() != ""].reset_index(drop=True)
        df['label_name'] = df['label'].map({0: 'Human', 1: 'ChatGPT'})
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        df['char_count'] = df['text'].apply(lambda x: len(x))
        df['punct_count'] = df['text'].apply(lambda x: len([c for c in x if c in string.punctuation]))
        return df, None
    except Exception as e:
        return None, str(e)

# ========================
# CACHE: Hitung top kata (berat, cache dulu)
# ========================
@st.cache_data
def get_top_words(df, label, n=20):
    stopwords = {'the','a','an','and','or','but','in','on','at','to','of','for',
                 'is','it','its','was','are','be','been','with','that','this',
                 'i','you','he','she','they','we','my','your','his','her','their',
                 'have','has','not','so','as','if','by','from','do','can','will'}
    text = " ".join(df[df['label']==label]['text'].astype(str)).lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stopwords]
    return Counter(words).most_common(n)

model, vectorizer = load_model()

# Preprocessing prediksi
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def predict_text(text):
    cleaned = clean_text(text)
    tfidf_feat = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf_feat)[0]
    prob = model.predict_proba(tfidf_feat)[0]
    return prediction, prob

# Sidebar
menu = st.sidebar.selectbox("Menu", ["Home (Detection)", "EDA", "Model Info"])

# ========================
# 1. HOME
# ========================
if menu == "Home (Detection)":
    st.title("🤖 AI vs Human Text Detection")
    st.write("Enter text [English] to detect whether it was created by AI or humans.")

    text = st.text_area("Text Input:", height=200)

    if st.button("Deteksi"):
        if text.strip() == "":
            st.warning("Enter the text first.")
        else:
            prediction, prob = predict_text(text)
            if prediction == 1:
                st.error("⚠️ AI Generated")
            else:
                st.success("✅ Human Written")
            st.write(f"Confidence: {max(prob)*100:.2f}%")

# ========================
# 2. EDA
# ========================
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    df, error = load_dataset()

    if error:
        st.error(f"Dataset gagal dimuat: {error}")
    else:
        st.success(f"Dataset berhasil dimuat: {len(df):,} sampel")

        # Info Umum
        st.subheader("📌 Informasi Umum Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sampel", f"{len(df):,}")
        col2.metric("Sampel Human", f"{len(df[df['label']==0]):,}")
        col3.metric("Sampel ChatGPT", f"{len(df[df['label']==1]):,}")

        st.write("**5 baris pertama data:**")
        st.dataframe(df[['text', 'label_name', 'word_count', 'char_count', 'punct_count']].head())

        # Distribusi Label
        st.subheader("📊 Distribusi Label")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        label_counts = df['label_name'].value_counts()
        bars = ax1.bar(label_counts.index, label_counts.values,
                       color=['#4C9BE8', '#E8674C'], width=0.5, edgecolor='none')
        for bar, val in zip(bars, label_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                     f'{val:,}', ha='center', va='bottom', fontsize=10)
        ax1.set_ylabel("Jumlah Sampel")
        ax1.set_title("Distribusi Label")
        ax1.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig1)

        # Distribusi Panjang Teks
        st.subheader("📊 Distribusi Panjang Teks (Karakter)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for label, color, name in [(0, '#4C9BE8', 'Human'), (1, '#E8674C', 'ChatGPT')]:
            ax2.hist(df[df['label']==label]['char_count'], bins=60, alpha=0.6,
                     color=color, label=name)
        ax2.set_xlim(0, 5000)
        ax2.set_xlabel("Jumlah Karakter")
        ax2.set_ylabel("Frekuensi")
        ax2.set_title("Distribusi Panjang Teks")
        ax2.legend()
        ax2.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig2)

        # Distribusi Jumlah Kata
        st.subheader("📊 Distribusi Jumlah Kata")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        for label, color, name in [(0, '#4C9BE8', 'Human'), (1, '#E8674C', 'ChatGPT')]:
            ax3.hist(df[df['label']==label]['word_count'], bins=60, alpha=0.6,
                     color=color, label=name)
        ax3.set_xlim(0, 1000)
        ax3.set_xlabel("Jumlah Kata")
        ax3.set_ylabel("Frekuensi")
        ax3.set_title("Distribusi Jumlah Kata")
        ax3.legend()
        ax3.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig3)

        # Boxplot
        st.subheader("📊 Perbandingan Statistik Fitur per Label")
        fig4, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, title in zip(axes,
                                   ['word_count', 'punct_count'],
                                   ['Jumlah Kata', 'Jumlah Tanda Baca']):
            data_plot = [df[df['label']==0][col], df[df['label']==1][col]]
            bp = ax.boxplot(data_plot, labels=['Human', 'ChatGPT'],
                            patch_artist=True, widths=0.4,
                            medianprops=dict(color='white', linewidth=2))
            bp['boxes'][0].set_facecolor('#4C9BE8')
            bp['boxes'][1].set_facecolor('#E8674C')
            ax.set_title(title)
            ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)

        # Rata-rata statistik
        st.subheader("📋 Rata-rata Statistik per Label")
        stats = df.groupby('label_name')[['word_count', 'char_count', 'punct_count']].mean().round(2)
        stats.columns = ['Rata-rata Kata', 'Rata-rata Karakter', 'Rata-rata Tanda Baca']
        st.dataframe(stats)

        # Top kata (di-cache supaya tidak berat)
        st.subheader("📊 Top 20 Kata — Human")
        human_freq = get_top_words(df, label=0)
        wdf_h = pd.DataFrame(human_freq, columns=['Kata', 'Frekuensi'])
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        ax5.barh(wdf_h['Kata'][::-1], wdf_h['Frekuensi'][::-1], color='#4C9BE8', edgecolor='none')
        ax5.set_xlabel("Frekuensi")
        ax5.set_title("Top 20 Kata — Human")
        ax5.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig5)

        st.subheader("📊 Top 20 Kata — ChatGPT")
        gpt_freq = get_top_words(df, label=1)
        wdf_g = pd.DataFrame(gpt_freq, columns=['Kata', 'Frekuensi'])
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        ax6.barh(wdf_g['Kata'][::-1], wdf_g['Frekuensi'][::-1], color='#E8674C', edgecolor='none')
        ax6.set_xlabel("Frekuensi")
        ax6.set_title("Top 20 Kata — ChatGPT")
        ax6.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig6)

# ========================
# 3. MODEL INFO
# ========================
elif menu == "Model Info":
    st.title("🧠 Model Information")

    st.subheader("Model yang Digunakan")
    st.write("**Logistic Regression + TF-IDF + Feature Engineering**")

    st.subheader("Arsitektur Fitur")
    st.write("""
    Model menggunakan kombinasi tiga jenis fitur:
    - **TF-IDF** (5000 fitur) — representasi statistik kata dalam dokumen
    - **Word Count** — jumlah kata dalam teks
    - **Punct Count** — jumlah tanda baca dalam teks
    """)

    st.subheader("Hasil Evaluasi Model")
    eval_data = {
        'Metrik': ['Accuracy', 'Precision (Human)', 'Recall (Human)', 'F1-Score (Human)',
                   'Precision (ChatGPT)', 'Recall (ChatGPT)', 'F1-Score (ChatGPT)'],
        'Nilai': ['96%', '97%', '95%', '96%', '95%', '97%', '96%']
    }
    st.dataframe(pd.DataFrame(eval_data))

    st.subheader("Kelebihan Model")
    st.write("""
    - Cepat dan efisien untuk data teks skala besar
    - Interpretable — bobot fitur dapat dianalisis
    - Performa tinggi (96%) pada data original
    - Ringan dan mudah di-deploy
    """)

    st.subheader("Keterbatasan Model")
    st.write("""
    - Bergantung pada pola permukaan (surface pattern)
    - Performa menurun pada teks hasil paraphrase atau editing manual
    - Tidak memahami konteks semantik secara mendalam
    - Belum mendukung bahasa selain Inggris
    """)

    st.subheader("Saran Pengembangan")
    st.write("""
    - Menggunakan model transformer seperti BERT atau RoBERTa
    - Menambahkan data augmentasi paraphrase saat training
    - Mendukung multi-bahasa
    """)