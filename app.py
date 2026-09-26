"""
app.py — NLP Sentiment & Emotion Analyzer
A Streamlit app for sentiment, emotion, keyword, and readability analysis.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import time
import json
from io import StringIO

import pandas as pd
import streamlit as st

# ---- NLP libraries ----
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


# =========================================================
# NLTK setup
# =========================================================
@st.cache_resource
def _setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

_setup_nltk()
STOPWORDS = set(stopwords.words("english"))


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Custom CSS
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
.hero {
    background: linear-gradient(120deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 12px 45px rgba(99,102,241,0.35);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon {
    background: rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 16px;
    backdrop-filter: blur(10px);
}
.hero h1 {
    color: #fff;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1rem;
    margin: 0.3rem 0 0 0;
}
.section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #e2e8f0;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 1.75rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.kpi-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.4);
}
.kpi-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.kpi-value {
    color: #f1f5f9;
    font-size: 1.85rem;
    font-weight: 800;
    margin-top: 0.5rem;
    letter-spacing: -0.5px;
}
.pill {
    display: inline-block;
    padding: 0.5rem 1.25rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.3px;
}
.pill-positive { background: rgba(16,185,129,0.18); color: #6ee7b7; border: 1px solid #10b981; }
.pill-negative { background: rgba(239,68,68,0.18);  color: #fca5a5; border: 1px solid #ef4444; }
.pill-neutral  { background: rgba(148,163,184,0.18);color: #cbd5e1; border: 1px solid #64748b; }
.chip {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    padding: 0.4rem 0.9rem;
    margin: 0.25rem 0.35rem 0.25rem 0;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 600;
    box-shadow: 0 3px 10px rgba(99,102,241,0.35);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    border-right: 1px solid rgba(99,102,241,0.2);
}
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    box-shadow: 0 8px 25px rgba(139,92,246,0.55);
}
.stTextArea textarea {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SVG icon helper
# =========================================================
def icon(name, size=20, color="#c7d2fe"):
    paths = {
        "brain":    '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2z"/>',
        "smile":    '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>',
        "heart":    '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
        "tag":      '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
        "book":     '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
        "globe":    '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
        "zap":      '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "sparkles": '<path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1m0-12.8l-2.1 2.1M7.7 16.3l-2.1 2.1"/>',
        "chart":    '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
        "upload":   '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
        "doc":      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
    }
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths.get(name, "")}</svg>'
    )


# =========================================================
# NLP core functions
# =========================================================
def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.15:
        label = "Positive"
    elif polarity < -0.15:
        label = "Negative"
    else:
        label = "Neutral"
    return {
        "label": label,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
    }


@st.cache_resource(show_spinner="Loading emotion model (first run only)...")
def load_emotion_pipeline():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
    )


EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "love"]

def analyze_emotion(text: str):
    pipe = load_emotion_pipeline()
    result = pipe(text, EMOTION_LABELS, multi_label=False)
    scores = dict(zip(result["labels"], [round(s, 3) for s in result["scores"]]))
    return result["labels"][0], round(result["scores"][0], 3), scores


def extract_keywords(text: str, top_n: int = 8):
    sentences = [s for s in sent_tokenize(text) if len(s.split()) > 3]
    if len(sentences) < 2:
        words = [
            w.lower() for w in word_tokenize(text)
            if w.isalnum() and w.lower() not in STOPWORDS and len(w) > 2
        ]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
    tfidf = vec.fit_transform(sentences)
    scores = tfidf.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: -x[1])
    return [(t, round(float(s), 4)) for t, s in ranked[:top_n]]


def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    return max(count, 1)


def analyze_readability(text: str) -> dict:
    sentences = sent_tokenize(text)
    words = [w for w in word_tokenize(text) if w.isalnum()]
    if not sentences or not words:
        return {"score": 0.0, "grade": "N/A", "words": 0, "sentences": 0, "avg_len": 0.0}
    syllables = sum(_count_syllables(w) for w in words)
    n_words = len(words)
    n_sents = len(sentences)
    score = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (syllables / n_words)
    score = max(0.0, min(100.0, score))
    if score >= 90:
        grade = "Very Easy (5th grade)"
    elif score >= 80:
        grade = "Easy (6th grade)"
    elif score >= 70:
        grade = "Fairly Easy (7th grade)"
    elif score >= 60:
        grade = "Standard (8-9th grade)"
    elif score >= 50:
        grade = "Fairly Difficult (10-12th)"
    elif score >= 30:
        grade = "Difficult (College)"
    else:
        grade = "Very Difficult (College+)"
    return {
        "score": round(score, 2),
        "grade": grade,
        "words": n_words,
        "sentences": n_sents,
        "avg_len": round(n_words / n_sents, 2),
    }


def analyze(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")
    sentiment = analyze_sentiment(text)
    emotion, conf, all_emotions = analyze_emotion(text)
    keywords = extract_keywords(text)
    readability = analyze_readability(text)
    try:
        lang = TextBlob(text).detect_language()
    except Exception:
        lang = "en"
    return {
        "text": text,
        "language": lang,
        "sentiment": sentiment,
        "emotion": {"top": emotion, "confidence": conf, "scores": all_emotions},
        "keywords": keywords,
        "readability": readability,
    }


# =========================================================
# Hero header
# =========================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-icon">{icon("brain", 46, "#ffffff")}</div>
    <div>
        <h1>NLP Sentiment & Emotion Analyzer</h1>
        <p>Transformer-powered text analysis · Sentiment · Emotion · Keywords · Readability</p>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown(f"""
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">
    {icon("sparkles", 22, "#a5b4fc")}
    <span style="color:#e2e8f0;font-size:1.1rem;font-weight:700;">Analysis Mode</span>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Choose input type",
    ["Single Text", "Batch (CSV)"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="display:flex;align-items:center;gap:0.5rem;color:#94a3b8;font-size:0.9rem;">
    {icon("zap", 16, "#a5b4fc")}
    <span><b>Models</b>: TextBlob · BART-MNLI · TF-IDF</span>
</div>
<br>
<div style="display:flex;align-items:center;gap:0.5rem;color:#94a3b8;font-size:0.9rem;">
    {icon("globe", 16, "#a5b4fc")}
    <span>Runs fully on CPU · No API keys needed</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Renderers
# =========================================================
def render_single_result(result: dict):
    sent = result["sentiment"]
    emo = result["emotion"]
    read = result["readability"]
    kws = result["keywords"]

    pill_class = {
        "Positive": "pill-positive",
        "Negative": "pill-negative",
        "Neutral":  "pill-neutral",
    }[sent["label"]]

    st.markdown(
        f'<div class="section-title">{icon("chart", 22)}Analysis Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">{icon("smile", 16)}Sentiment</div>
            <div style="margin-top:0.7rem;">
                <span class="pill {pill_class}">{sent['label']}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{icon("heart", 16)}Top Emotion</div>
            <div class="kpi-value">{emo['top'].capitalize()}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{icon("zap", 16)}Polarity</div>
            <div class="kpi-value">{sent['polarity']:+.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{icon("book", 16)}Readability</div>
            <div class="kpi-value">{read['score']:.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f'<div class="section-title">{icon("heart", 20)}Emotion Breakdown</div>',
            unsafe_allow_html=True,
        )
        emo_df = pd.DataFrame(
            {"Emotion": list(emo["scores"].keys()),
             "Confidence": list(emo["scores"].values())}
        ).set_index("Emotion")
        st.bar_chart(emo_df, height=260)

    with c2:
        st.markdown(
            f'<div class="section-title">{icon("book", 20)}Text Statistics</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Words</div>
                <div class="kpi-value">{read['words']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Sentences</div>
                <div class="kpi-value">{read['sentences']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            f"**Subjectivity:** `{sent['subjectivity']:.2f}` "
            f"&nbsp;·&nbsp; **Avg sentence length:** `{read['avg_len']}` words"
        )
        st.markdown(f"**Reading level:** {read['grade']}")

    st.markdown(
        f'<div class="section-title">{icon("tag", 20)}Top Keywords</div>',
        unsafe_allow_html=True,
    )
    if kws:
        chips = "".join(
            f'<span class="chip">{kw} &nbsp;·&nbsp; {score:.3f}</span>'
            for kw, score in kws
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.info("No keywords extracted.")

    st.markdown(
        f'<div class="section-title">{icon("globe", 20)}Detected Language</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**`{result['language'].upper()}`**")

    with st.expander("View raw JSON output"):
        st.json(result)


def render_batch_results(results):
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"text": r["text"][:80], "error": r["error"]})
            continue
        rows.append({
            "text": r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
            "sentiment": r["sentiment"]["label"],
            "polarity": r["sentiment"]["polarity"],
            "subjectivity": r["sentiment"]["subjectivity"],
            "top_emotion": r["emotion"]["top"],
            "emotion_conf": r["emotion"]["confidence"],
            "readability": r["readability"]["score"],
            "language": r["language"],
        })
    df = pd.DataFrame(rows)

    st.markdown(
        f'<div class="section-title">{icon("chart", 22)}Batch Results</div>',
        unsafe_allow_html=True,
    )

    total = len([r for r in results if "error" not in r])
    pos = sum(1 for r in results if "error" not in r and r["sentiment"]["label"] == "Positive")
    neg = sum(1 for r in results if "error" not in r and r["sentiment"]["label"] == "Negative")
    neu = total - pos - neg

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Processed</div>
            <div class="kpi-value">{total}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Positive</div>
            <div class="kpi-value">{pos}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Neutral</div>
            <div class="kpi-value">{neu}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Negative</div>
            <div class="kpi-value">{neg}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True, height=400)

    st.markdown(
        f'<div class="section-title">{icon("chart", 20)}Sentiment Distribution</div>',
        unsafe_allow_html=True,
    )
    dist = pd.DataFrame({"Count": [pos, neu, neg]},
                        index=["Positive", "Neutral", "Negative"])
    st.bar_chart(dist, height=280)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        data=csv,
        file_name="nlp_batch_results.csv",
        mime="text/csv",
    )


# =========================================================
# Main UI
# =========================================================
SAMPLE = (
    "I absolutely love this new project! The team has been incredibly supportive "
    "and the work feels meaningful. However, the deadline is approaching fast and "
    "I'm a little worried about finishing on time. Overall, I'm excited about what "
    "we're building and grateful for the opportunity."
)

if mode == "Single Text":
    st.markdown(
        f'<div class="section-title">{icon("doc", 22)}Your Text</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("Load sample", use_container_width=True):
            st.session_state["text_input"] = SAMPLE

    text = st.text_area(
        "Paste or type text to analyze",
        value=st.session_state.get("text_input", ""),
        height=200,
        label_visibility="collapsed",
        placeholder="Type something like: 'I love this product, but the delivery was slow...'",
    )

    if st.button("Analyze Text", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Running analysis..."):
                t0 = time.time()
                try:
                    result = analyze(text)
                    elapsed = time.time() - t0
                    st.caption(f"Analyzed in {elapsed:.2f}s")
                    render_single_result(result)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

else:
    st.markdown(
        f'<div class="section-title">{icon("upload", 22)}Upload CSV</div>',
        unsafe_allow_html=True,
    )
    st.info("CSV must have a `text` column. Each row is analyzed independently.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            if "text" not in df_in.columns:
                st.error("CSV must contain a column named `text`.")
            else:
                texts = df_in["text"].dropna().astype(str).tolist()
                st.success(f"Loaded {len(texts)} rows.")
                st.dataframe(df_in.head(), use_container_width=True)

                if st.button("Analyze Batch", use_container_width=True):
                    progress = st.progress(0.0)
                    status = st.empty()
                    results = []
                    for i, t in enumerate(texts):
                        status.markdown(f"**Processing {i+1} / {len(texts)}**")
                        try:
                            results.append(analyze(t))
                        except Exception as e:
                            results.append({"text": t, "error": str(e)})
                        progress.progress((i + 1) / len(texts))
                    status.empty()
                    progress.empty()
                    render_batch_results(results)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

st.caption("Built with Streamlit · TextBlob · HuggingFace Transformers · scikit-learn · NLTK")
