streamlit run app.py

import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------------
st.set_page_config(
    page_title="OGGen AI Transformer Suite",
    page_icon="✨",
    layout="wide"
)

# -----------------------------------------------------
# CUSTOM THEME (Maroon - Cream - Orange - Brown)
# -----------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #fdf6f2; /* soft cream */
}

.title {
    font-size: 48px;
    text-align: center;
    font-weight: 900;
    color: #7a0010; /* deep maroon */
    margin-bottom: -10px;
    animation: fadeIn 2s;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #8a4b2d; /* brown */
    margin-bottom: 25px;
}

.card-btn {
    background-color: #ffe8df;
    border-radius: 18px;
    padding: 25px;
    border: 3px solid #7a0010;
    transition: 0.2s ease-in-out;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(122,0,16,0.2);
}

.card-btn:hover {
    background-color: #ffd8cc;
    transform: scale(1.02);
    cursor: pointer;
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    color: #7a0010;
}

.card-desc {
    font-size: 14px;
    color: #c25100; /* warm orange */
}

.inner-card {
    background: #fff4ed;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 2px 10px rgba(122, 0, 16, 0.15);
    margin-top: 20px;
}

@keyframes fadeIn {
    0% {opacity:0;}
    100% {opacity:1;}
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.markdown('<div class="title">✨ OGGen AI – Interactive NLP Suite ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose a task below — everything is powered by transformers!</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# LOAD ALL MODELS
# -----------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "generator": pipeline("text-generation", model="gpt2"),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
        "sentiment": pipeline("sentiment-analysis"),
        "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
        "qa": pipeline("question-answering"),
        "translate": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "para": pipeline("text2text-generation", model="t5-small"),
        "grammar": pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1"),
        "embed": SentenceTransformer("all-MiniLM-L6-v2")
    }

models = load_models()

# -----------------------------------------------------
# BUTTON-BASED CARD NAVIGATION
# -----------------------------------------------------

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8, col9 = st.columns(3)

selected = None

if col1.button("📝 Text Generation", key="tg"):
    selected = "Text Generation"
if col2.button("📚 Summarization", key="sum"):
    selected = "Summarization"
if col3.button("😊 Sentiment", key="sent"):
    selected = "Sentiment Analysis"

if col4.button("🏷 NER", key="ner"):
    selected = "NER"
if col5.button("❓ Question Answering", key="qa"):
    selected = "QA"
if col6.button("🌍 Translation", key="trans"):
    selected = "Translation"

if col7.button("♻️ Paraphrasing", key="para"):
    selected = "Paraphrasing"
if col8.button("🔑 Keyword Extraction", key="kw"):
    selected = "Keyword"
if col9.button("✍️ Grammar Correction", key="gram"):
    selected = "Grammar"

st.write("")  # space

# -----------------------------------------------------
# FEATURE CARD SHOWS AFTER BUTTON CLICK
# -----------------------------------------------------

if selected:

    st.markdown(f"<div class='inner-card'><h3 style='color:#7a0010;'>{selected}</h3>", unsafe_allow_html=True)

    # ---------------------------------------
    # TEXT GENERATION
    # ---------------------------------------
    if selected == "Text Generation":
        text = st.text_area("Enter your prompt:", "BJP is")
        if st.button("Generate Text"):
            out = models["generator"](text, max_length=300)
            st.success(out[0]["generated_text"])

    # ---------------------------------------
    # SUMMARIZATION
    # ---------------------------------------
    if selected == "Summarization":
        text = st.text_area("Enter text to summarize:")
        if st.button("Summarize"):
            summary = models["summarizer"](text, max_length=50, min_length=10, do_sample=False)
            st.success(summary[0]["summary_text"])

    # ---------------------------------------
    # SENTIMENT ANALYSIS
    # ---------------------------------------
    if selected == "Sentiment Analysis":
        text = st.text_area("Enter text:")
        if st.button("Analyze Sentiment"):
            st.success(models["sentiment"](text))

    # ---------------------------------------
    # NER
    # ---------------------------------------
    if selected == "NER":
        text = st.text_area("Enter text:")
        if st.button("Extract Entities"):
            st.success(models["ner"](text))

    # ---------------------------------------
    # QUESTION ANSWERING
    # ---------------------------------------
    if selected == "QA":
        q = st.text_input("Enter your question:")
        context = st.text_area("Enter context:")
        if st.button("Get Answer"):
            st.success(models["qa"](question=q, context=context))

    # ---------------------------------------
    # TRANSLATION
    # ---------------------------------------
    if selected == "Translation":
        text = st.text_input("Enter English text:")
        if st.button("Translate to French"):
            st.success(models["translate"](text)[0]["translation_text"])

    # ---------------------------------------
    # PARAPHRASING
    # ---------------------------------------
    if selected == "Paraphrasing":
        text = st.text_input("Enter sentence:")
        if st.button("Paraphrase"):
            st.success(models["para"]("paraphrase: " + text)[0]["generated_text"])

    # ---------------------------------------
    # KEYWORD EXTRACTION (NER-Based)
    # ---------------------------------------
    if selected == "Keyword":
        text = st.text_area("Enter a sentence:")
        if st.button("Extract Keywords"):
            st.success(models["ner"](text))

    # ---------------------------------------
    # GRAMMAR CORRECTION
    # ---------------------------------------
    if selected == "Grammar":
        text = st.text_input("Enter incorrect sentence:")
        if st.button("Correct Grammar"):
            st.success(models["grammar"](text)[0]["generated_text"])

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.markdown(
    """
    <br><center>
    <p style='color:#7a0010; font-size:18px; font-weight:700;'>✨ Thank you for using OGGen AI! ✨</p>
    </center>
    """,
    unsafe_allow_html=True
)
