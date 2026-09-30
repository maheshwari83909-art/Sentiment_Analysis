import streamlit as st
from PIL import Image
import os
import pytesseract
tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
from transformers import pipeline
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* ===== MAIN APP ===== */

.stApp {
    background: #0f172a;
    color: #e2e8f0;
}

/* ===== TITLE ===== */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
}

.subtitle {
    font-size: 17px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* ===== CARDS ===== */

.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

/* ===== TEXT AREA ===== */

textarea {
    background-color: #f8fafc !important;
    color: #111827 !important;
    border: 2px solid #475569 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    padding: 14px !important;
}

/* Text area placeholder */

textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* ===== FILE UPLOADER ===== */

[data-testid="stFileUploader"] {
    background: #1e293b !important;
    border: 2px dashed #64748b !important;
    border-radius: 15px !important;
    padding: 15px !important;
}

/* File uploader text */

[data-testid="stFileUploader"] label {
    color: #f8fafc !important;
}

[data-testid="stFileUploader"] section {
    background: #1e293b !important;
}

[data-testid="stFileUploader"] small {
    color: #cbd5e1 !important;
}

/* Browse files button */

[data-testid="stFileUploader"] button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

/* ===== RADIO BUTTON ===== */

[data-testid="stRadio"] label {
    color: #e2e8f0 !important;
}

/* ===== NORMAL TEXT ===== */

p, label, span {
    color: #e2e8f0;
}

/* ===== HEADINGS ===== */

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

/* ===== BUTTON ===== */

.stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: #1d4ed8 !important;
}

/* ===== METRIC ===== */

.metric-title {
    color: #94a3b8;
    font-size: 15px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: white;
}

/* ===== TEXT OUTPUT ===== */

.stTextArea textarea {
    background-color: #f8fafc !important;
    color: #111827 !important;
}

/* ===== SUCCESS / WARNING / ERROR ===== */

[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {
    background: #111827 !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_model = load_model()

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🧠 SentimentIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Text & Image Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("⚙️ Analysis Mode")

    mode = st.radio(
        "Select input",
        ["📝 Text Analysis", "📷 Image Analysis"]
    )

    st.markdown("---")

    st.info(
        "Upload text or an image containing text. "
        "The AI model will classify the sentiment."
    )

# ---------------- FUNCTIONS ----------------

def analyze_sentiment(text):

    result = sentiment_model(text[:512])[0]

    label = result["label"].lower()
    confidence = result["score"] * 100

    if "positive" in label:
        sentiment = "Positive"
    elif "negative" in label:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, confidence


def show_result(sentiment, confidence):

    if sentiment == "Positive":
        st.markdown(
            '<div class="result-positive">😊 POSITIVE</div>',
            unsafe_allow_html=True
        )

    elif sentiment == "Negative":
        st.markdown(
            '<div class="result-negative">😞 NEGATIVE</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="result-neutral">😐 NEUTRAL</div>',
            unsafe_allow_html=True
        )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="card">'
            f'<div class="metric-title">Detected Sentiment</div>'
            f'<div class="metric-value">{sentiment}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="card">'
            f'<div class="metric-title">Confidence</div>'
            f'<div class="metric-value">{confidence:.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="card">'
            f'<div class="metric-title">AI Model</div>'
            f'<div class="metric-value">RoBERTa</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ---------------- TEXT MODE ----------------

if mode == "📝 Text Analysis":

    st.subheader("📝 Enter Text")

    text = st.text_area(
        "Write your text below",
        height=180,
        placeholder=
        "Example: I really enjoyed this product. "
        "The quality is excellent!"
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):

        if not text.strip():

            st.warning("Please enter some text.")

        else:

            with st.spinner("AI is analyzing your text..."):

                sentiment, confidence = analyze_sentiment(text)

            st.markdown("## 📊 Analysis Result")

            show_result(sentiment, confidence)

            st.markdown("### 📄 Analyzed Text")

            st.markdown(
                f'<div class="card">{text}</div>',
                unsafe_allow_html=True
            )


# ---------------- IMAGE MODE ----------------

else:

    st.subheader("📷 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload an image containing text",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 🖼️ Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### 🔎 OCR Preview")

            if st.button(
                "✨ Extract & Analyze",
                use_container_width=True
            ):

                with st.spinner("Reading image..."):

                    extracted_text = pytesseract.image_to_string(
                        image
                    )

                if not extracted_text.strip():

                    st.error(
                        "No readable text found in this image."
                    )

                else:

                    st.success("Text successfully extracted!")

                    st.text_area(
                        "Extracted Text",
                        extracted_text,
                        height=180
                    )

                    with st.spinner(
                        "Analyzing sentiment..."
                    ):

                        sentiment, confidence = (
                            analyze_sentiment(extracted_text)
                        )

                    st.markdown("## 📊 Analysis Result")

                    show_result(
                        sentiment,
                        confidence
                    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "SentimentIQ • AI-based Sentiment Analysis "
    "using RoBERTa + OCR"
)
