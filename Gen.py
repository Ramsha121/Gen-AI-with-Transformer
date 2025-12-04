import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="OGGen AI Transformer Suite",
    page_icon="✨",
    layout="wide"
)

# ================== STYLING ==================
st.markdown("""
<style>
    body { background-color: #f8f3ef; }
    .title { font-size: 45px; font-weight: 800; color:#5b0011; text-align:center; }
    .subtitle { font-size: 20px; text-align:center; margin-top:-10px; color:#3b2f2f; }
    .card {
        background-color: #fff7f5;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0px 2px 10px rgba(91,0,17,0.15);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ================== TITLE ==================
st.markdown("<div class='title'>✨ OGGen AI – Interactive NLP Suite ✨</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smarter UI • Theme-Based • Fully Button Interactive</div>", unsafe_allow_html=True)

# ================== SIDEBAR NAVIGATION ==================
menu = st.sidebar.selectbox(
    "Choose Tool:",
    [
        "Home",
        "Text Generation",
        "Summarization",
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Question Answering",
        "Translation",
        "Paraphrasing",
        "Keyword Extraction",
        "Grammar Correction",
        "Text Similarity"
    ]
)

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    return {
        "generator": pipeline("text-generation", model="gpt2"),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
        "sentiment": pipeline("sentiment-analysis"),
        "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
        "qa": pipeline("question-answering"),
        "translate_en_fr": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "translate_en_de": pipeline("translation", model="Helsinki-NLP/opus-mt-en-de"),
        "translate_en_hi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
        "para": pipeline("text2text-generation", model="t5-small"),
        "grammar": pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1"),
        "embed": SentenceTransformer("all-MiniLM-L6-v2")
    }

models = load_models()

# ========================== HOME ==========================
if menu == "Home":
    st.markdown("""
    <div class='card'>
        <h3 style='color:#5b0011;'>Welcome to OGGen AI ✨</h3>
        <p>This is an enhanced, interactive NLP application with buttons for all major operations.
        Explore text generation, summarization, translation, QA, NER, sentiment analysis and more.  
        Designed with a maroon–cream aesthetic for a premium feel.</p>
        <h4 style='color:#5b0011;'>Enjoy your experience!</h4>
    </div>
    """, unsafe_allow_html=True)


# ========================== TEXT GENERATION ==========================
if menu == "Text Generation":
    st.markdown("<div class='card'><h3>📝 Text Generation</h3>", unsafe_allow_html=True)
    prompt = st.text_area("Enter a prompt:", "BJP is")

    length = 200
    if st.button("Generate 200 tokens"):
        length = 200
    if st.button("Generate 400 tokens"):
        length = 400

    if st.button("Generate Text"):
        result = models["generator"](prompt, max_length=length)
        st.success(result[0]["generated_text"])
    st.markdown("</div>", unsafe_allow_html=True)


# ========================== SUMMARIZATION ==========================
if menu == "Summarization":
    st.markdown("<div class='card'><h3>📚 Summarization</h3>", unsafe_allow_html=True)
    text = st.text_area("Enter text:")

    if st.button("Short Summary (30 words)"):
        summary = models["summarizer"](text, max_length=30, min_length=10, do_sample=False)
        st.success(summary[0]["summary_text"])

    if st.button("Medium Summary (50 words)"):
        summary = models["summarizer"](text, max_length=50, min_length=20, do_sample=False)
        st.success(summary[0]["summary_text"])

    if st.button("Long Summary (100 words)"):
        summary = models["summarizer"](text, max_length=100, min_length=40, do_sample=False)
        st.success(summary[0]["summary_text"])

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== SENTIMENT ==========================
if menu == "Sentiment Analysis":
    st.markdown("<div class='card'><h3>😊 Sentiment Analysis</h3>", unsafe_allow_html=True)
    text = st.text_area("Enter text:")

    if st.button("Analyze Sentiment"):
        result = models["sentiment"](text)
        st.success(result)
    st.markdown("</div>", unsafe_allow_html=True)


# ========================== NER ==========================
if menu == "Named Entity Recognition":
    st.markdown("<div class='card'><h3>🏷 NER (Keyword Extraction)</h3>", unsafe_allow_html=True)
    text = st.text_area("Enter text:")

    if st.button("Extract Entities"):
        result = models["ner"](text)
        st.success(result)

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== QA ==========================
if menu == "Question Answering":
    st.markdown("<div class='card'><h3>❓ Question Answering</h3>", unsafe_allow_html=True)
    question = st.text_input("Enter question:")
    context = st.text_area("Enter context:")

    if st.button("Get Answer"):
        result = models["qa"](question=question, context=context)
        st.success(result)

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== TRANSLATION ==========================
if menu == "Translation":
    st.markdown("<div class='card'><h3>🌍 Translation</h3>", unsafe_allow_html=True)
    text = st.text_input("Enter English text:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("English → French"):
            st.success(models["translate_en_fr"](text)[0]["translation_text"])

    with col2:
        if st.button("English → German"):
            st.success(models["translate_en_de"](text)[0]["translation_text"])

    with col3:
        if st.button("English → Hindi"):
            st.success(models["translate_en_hi"](text)[0]["translation_text"])

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== PARAPHRASING ==========================
if menu == "Paraphrasing":
    st.markdown("<div class='card'><h3>♻️ Paraphrasing</h3>", unsafe_allow_html=True)
    text = st.text_input("Enter text:")

    if st.button("Simple Paraphrase"):
        st.success(models["para"]("paraphrase: " + text)[0]["generated_text"])

    if st.button("More Creative"):
        st.success(models["para"]("paraphrase it creatively: " + text)[0]["generated_text"])

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== GRAMMAR ==========================
if menu == "Grammar Correction":
    st.markdown("<div class='card'><h3>✍️ Grammar Correction</h3>", unsafe_allow_html=True)
    text = st.text_input("Enter incorrect sentence:")

    if st.button("Correct Grammar"):
        st.success(models["grammar"](text)[0]["generated_text"])

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== SIMILARITY ==========================
if menu == "Text Similarity":
    st.markdown("<div class='card'><h3>🔗 Text Similarity</h3>", unsafe_allow_html=True)

    s1 = st.text_input("Sentence 1:")
    s2 = st.text_input("Sentence 2:")

    if st.button("Calculate Similarity"):
        e1 = models["embed"].encode(s1, convert_to_tensor=True)
        e2 = models["embed"].encode(s2, convert_to_tensor=True)
        score = util.pytorch_cos_sim(e1, e2)
        st.success(score)

    st.markdown("</div>", unsafe_allow_html=True)


# ========================== FOOTER ==========================
st.markdown("""
<br><center><p style='color:#5b0011;font-weight:600;font-size:18px;'>
✨ Thank you for using OGGen AI! ✨
</p></center>
""", unsafe_allow_html=True)
