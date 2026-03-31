
import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(page_title="COVID Sentiment Analyzer", page_icon="🦠", layout="wide")

SENTIMENT_STYLE = {
    "Extremely Positive": {"bg": "#d4edda", "color": "#155724", "emoji": "🌟"},
    "Positive":           {"bg": "#d1ecf1", "color": "#0c5460", "emoji": "😊"},
    "Neutral":            {"bg": "#fff3cd", "color": "#856404", "emoji": "😐"},
    "Negative":           {"bg": "#f8d7da", "color": "#721c24", "emoji": "😟"},
    "Extremely Negative": {"bg": "#f5c6cb", "color": "#491217", "emoji": "😡"},
}

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def train_model():
    df = pd.read_csv("Corona_NLP_train.csv", usecols=["OriginalTweet", "Sentiment"])
    df["clean"] = df["OriginalTweet"].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean"])
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, C=5, solver="lbfgs", multi_class="multinomial")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return model, vectorizer, acc, report, cm, model.classes_, df

st.markdown("## 🦠 COVID-19 Sentiment Analyzer")
st.markdown("---")

with st.spinner("Training model... please wait"):
    model, vectorizer, acc, report, cm, classes, df = train_model()

st.success(f"✅ Model ready! Accuracy: {acc*100:.2f}%")

tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Metrics", "☁️ Word Cloud"])

with tab1:
    st.markdown("### Type any sentence to predict its sentiment")
    user_input = st.text_area("Enter text", placeholder="e.g. Vaccines are giving us hope!", height=120)
    if st.button("🔍 Analyse"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            style = SENTIMENT_STYLE.get(pred, {"bg":"#eee","color":"#333","emoji":"❓"})
            st.markdown(f"""
            <div style='background:{style["bg"]};color:{style["color"]};padding:1.5rem;
            border-radius:12px;text-align:center;font-size:1.5rem;font-weight:700'>
            {style["emoji"]} {pred}
            </div>""", unsafe_allow_html=True)
            st.markdown("#### Confidence")
            prob_df = pd.DataFrame({"Sentiment": classes, "Probability": proba}).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Sentiment"))
        else:
            st.warning("Please enter some text!")

with tab2:
    st.markdown("### Model Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Macro F1", f"{report['macro avg']['f1-score']*100:.2f}%")
    col3.metric("Macro Precision", f"{report['macro avg']['precision']*100:.2f}%")

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Per-Class Report")
    rows = [{"Sentiment": l, "Precision": f"{report[l]['precision']:.2f}",
             "Recall": f"{report[l]['recall']:.2f}", "F1": f"{report[l]['f1-score']:.2f}"}
            for l in classes if l in report]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Word Cloud by Sentiment")
    selected = st.selectbox("Choose a sentiment", list(SENTIMENT_STYLE.keys()))
    subset = df[df["Sentiment"] == selected]["clean"]
    text = " ".join(subset.dropna().tolist())
    wc = WordCloud(width=900, height=400, background_color="white", max_words=150).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
