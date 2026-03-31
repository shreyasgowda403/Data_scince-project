
import streamlit as st
import pickle
import numpy as np
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; background-color: #0d0d0d; color: #e8e0d0; }
#MainMenu, footer, header { visibility: hidden; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 3.2rem; line-height: 1.1; color: #f0e8d8; margin-bottom: 0.2rem; }
.hero-sub { font-size: 0.78rem; letter-spacing: 0.18em; color: #6b6560; text-transform: uppercase; margin-bottom: 2.5rem; }
textarea { background-color: #161616 !important; border: 1px solid #2a2a2a !important; border-radius: 4px !important; color: #e8e0d0 !important; font-size: 0.95rem !important; }
textarea:focus { border-color: #c8a96e !important; }
.stButton > button { background: #c8a96e; color: #0d0d0d; border: none; border-radius: 3px; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.65rem 2rem; width: 100%; }
.stButton > button:hover { background: #d4b87a; }
.result-card { margin-top: 2rem; padding: 1.8rem 2rem; border-radius: 4px; border-left: 3px solid; background: #161616; }
.result-label { font-family: 'DM Serif Display', serif; font-size: 2rem; margin-bottom: 0.3rem; }
.result-meta { font-size: 0.75rem; color: #6b6560; letter-spacing: 0.1em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Sentiment<br><em>Analyzer</em></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">COVID-19 Tweets · Logistic Regression · NLP</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, None
    except FileNotFoundError as e:
        return None, None, str(e)

model, vectorizer, load_error = load_model()

if load_error:
    st.error(f"Model files not found: {load_error}")
    st.stop()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

def predict(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(np.max(proba))
    return pred, round(confidence * 100, 1)

user_input = st.text_area(
    label="Text",
    placeholder="Type or paste a tweet here...",
    height=150,
    label_visibility="collapsed"
)

if st.button("Analyze →"):
    text = user_input.strip()
    if not text:
        st.error("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict(text)

        color_map = {
            "Extremely Positive": "#4CAF50",
            "Positive": "#6abf7b",
            "Neutral": "#c8a96e",
            "Negative": "#bf6a6a",
            "Extremely Negative": "#e53935"
        }
        emoji_map = {
            "Extremely Positive": "⬆⬆",
            "Positive": "＋",
            "Neutral": "◦",
            "Negative": "－",
            "Extremely Negative": "⬇⬇"
        }
        color = color_map.get(label, "#c8a96e")
        emoji = emoji_map.get(label, "◦")

        st.markdown(f"""
        <div class="result-card" style="border-color:{color}">
            <div class="result-label" style="color:{color}">{emoji} {label}</div>
            <div class="result-meta">Confidence · {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="font-size:0.72rem;color:#3d3a36;text-align:center;margin-top:3rem;">powered by your logistic regression model</div>', unsafe_allow_html=True)
