# 🧠 NLP Sentiment & Emotion Analyzer

A production-ready NLP web application that analyzes text for **sentiment**,
**emotion**, **keywords**, and **readability** — powered by HuggingFace
Transformers, TextBlob, and classical NLP techniques.

Built with **Streamlit** and deployable to **Streamlit Community Cloud** in
under 5 minutes.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Sentiment Analysis** | Positive / Neutral / Negative with polarity & subjectivity scores (TextBlob) |
| **Emotion Detection** | Zero-shot classification across 6 emotions: joy, sadness, anger, fear, surprise, love (BART-MNLI) |
| **Keyword Extraction** | Top-N keywords via TF-IDF over sentences |
| **Readability Score** | Flesch Reading Ease with grade-level classification |
| **Language Detection** | Detects language of input text |
| **Batch Mode** | Upload a CSV with a `text` column for bulk analysis |
| **CSV Export** | Download results as CSV |
| **Modern Dark UI** | Custom CSS theme, gradient hero, KPI cards, SVG icons |

---

## 📁 Project Structure

```
NLP-Sentiment-Analyzer/
├── app.py              # Complete app (UI + NLP logic)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

Single-file app — everything lives in `app.py` for easy deployment.

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/nlp-sentiment-analyzer.git
cd nlp-sentiment-analyzer
```

### 2. Create a virtual environment
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

> **First run note:** The BART-MNLI emotion model (~1.6 GB) downloads on first
> use. Subsequent runs load from cache and are fast.

---

## 🧪 How It Works

### 1. Sentiment Analysis — TextBlob
- Computes **polarity** (-1 to +1) and **subjectivity** (0 to 1)
- Threshold ±0.15 determines Positive / Negative / Neutral

### 2. Emotion Detection — Zero-Shot Classification
- Model: `facebook/bart-large-mnli`
- Classifies text into 6 candidate emotions without task-specific training
- Returns confidence score per emotion

### 3. Keyword Extraction — TF-IDF
- Splits text into sentences
- Applies TF-IDF vectorization (unigrams + bigrams)
- Ranks terms by summed TF-IDF weight
- Falls back to word frequency for single-sentence input

### 4. Readability — Flesch Reading Ease
Formula:
```
206.835 − 1.015 × (words / sentences) − 84.6 × (syllables / words)
```
Maps the score to a US grade-level label.

### 5. Language Detection — TextBlob
Uses TextBlob's built-in `detect_language()`.

---

## ☁️ Deployment

### Option A — Streamlit Community Cloud (Recommended, Free)

1. **Push your project to GitHub:**
   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/nlp-sentiment-analyzer.git
   git push -u origin main
   ```

2. **Go to** https://share.streamlit.io

3. **Sign in with GitHub** and click **"New app"**

4. **Configure:**
   - Repository: `YOUR_USERNAME/nlp-sentiment-analyzer`
   - Branch: `main`
   - Main file path: `app.py`

5. **(Optional)** In *Advanced settings*, set Python version to **3.10** or **3.11**.

6. **Click Deploy** — your app will be live at:
   ```
   https://YOUR_USERNAME-nlp-sentiment-analyzer.streamlit.app
   ```

> **Memory tip:** If Streamlit Cloud runs out of RAM with the BART model,
> swap it for a smaller one in `app.py`:
> ```python
> model="typeform/distilbert-base-uncased-mnli"  # ~270 MB
> ```

---

### Option B — Hugging Face Spaces (Free)

1. Go to https://huggingface.co/spaces → **Create new Space**
2. SDK: **Streamlit**, Hardware: **CPU basic** (free)
3. Upload `app.py`, `requirements.txt`, `README.md`
4. HF auto-installs and runs — your app appears at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/nlp-sentiment-analyzer
   ```

---

### Option C — Docker

Create a `Dockerfile`:
```dockerfile
