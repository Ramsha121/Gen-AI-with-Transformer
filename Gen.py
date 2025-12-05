import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import time

# --- Configuration and Initialization ---

st.set_page_config(
    page_title="OGGen AI Transformer Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTitle, .stHeader {
        color: #58a6ff;
    }
    .css-1d391kg, .css-1lcbmhc {
        background-color: #161b22;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: 1px solid #30363d;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
#   MODEL LOADERS (FIXED: Merged the two sections and removed duplicates)
# =====================================================================

@st.cache_resource(show_spinner="Loading GPT-2 Generator...")
def load_generator():
    return pipeline("text-generation", model="gpt2")

@st.cache_resource(show_spinner="Loading Summarizer...")
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner="Loading Sentiment Model...")
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource(show_spinner="Loading NER Model...")
def load_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource(show_spinner="Loading QA Model...")
def load_qa_model():
    return pipeline("question-answering")

@st.cache_resource(show_spinner="Loading Translation Model...")
def load_translator():
    try:
        return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    except:
        return pipeline("text2text-generation", model="t5-small")

@st.cache_resource(show_spinner="Loading Paraphrasing Model...")
def load_paraphraser():
    # Only one definition, using the model from the first section
    return pipeline(
        "text2text-generation",
        model="Vamsi/T5_Paraphrase_Paws"
    )

@st.cache_resource(show_spinner="Loading Grammar Corrector...")
def load_grammar_corrector():
    # Only one definition, using the model from the first section
    return pipeline(
        "text2text-generation",
        model="prithivida/grammar_error_correcter_v1"
    )

@st.cache_resource(show_spinner="Loading Similarity Model...")
def load_similarity_model():
    # Only one definition
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =====================================================================
#   UTILITY FUNCTIONS (UNCHANGED)
# =====================================================================

def run_text_generation(prompt, max_len):
    generator = load_generator()
    result = generator(prompt, max_length=max_len)
    return result[0]["generated_text"]

def run_summarization(text, max_len, min_len):
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]["summary_text"]

def run_sentiment_analysis(text):
    model = load_sentiment_model()
    return model(text)[0]

def run_ner(text):
    ner = load_ner_model()
    return ner(text)

def run_qa(question, context):
    qa = load_qa_model()
    return qa(question=question, context=context)

def run_translation(text):
    trans = load_translator()
    return trans(text)[0]["translation_text"]

def run_paraphrasing(text):
    para = load_paraphraser()
    # T5 models often require a prefix, ensure the correct one is used
    return para(f"paraphrase: {text}")[0]["generated_text"]

def run_grammar_correction(text):
    gc = load_grammar_corrector()
    return gc(text)[0]["generated_text"]

def run_text_similarity(text_a, text_b):
    model = load_similarity_model()
    a = model.encode(text_a, convert_to_tensor=True)
    b = model.encode(text_b, convert_to_tensor=True)
    return util.pytorch_cos_sim(a, b).item()

# =====================================================================
#   PLACEHOLDER UI FUNCTIONS (ADDED to prevent NameError)
#   *** REPLACE THESE WITH YOUR ACTUAL UI CODE ***
# =====================================================================

def page_text_generation():
    st.title("✍️ Text Generation (Placeholder)")
    st.info("Replace this with your actual Text Generation UI.")

def page_summarization():
    st.title("📝 Text Summarization (Placeholder)")
    st.info("Replace this with your actual Text Summarization UI.")

def page_sentiment_analysis():
    st.title("😊 Sentiment Analysis (Placeholder)")
    st.info("Replace this with your actual Sentiment Analysis UI.")

def page_ner():
    st.title("📍 Named Entity Recognition (NER) (Placeholder)")
    st.info("Replace this with your actual NER UI.")

def page_qa():
    st.title("❓ Question Answering (QA) (Placeholder)")
    st.info("Replace this with your actual QA UI.")

def page_translation():
    st.title("🌐 English to French Translation (Placeholder)")
    st.info("Replace this with your actual Translation UI.")

def page_paraphrase_grammar():
    st.title("✨ Text Refinement (Paraphrase/Grammar) (Placeholder)")
    st.info("Replace this with your actual Text Refinement UI.")

def page_text_similarity():
    st.title("🔢 Semantic Text Similarity (Placeholder)")
    st.info("Replace this with your actual Text Similarity UI.")


# =====================================================================
#   NAVIGATION + MODEL PRELOAD
# =====================================================================

st.sidebar.markdown("# **OGGen AI Hub**")
st.sidebar.markdown("Explore various NLP tasks powered by Hugging Face Transformers.")

# Define the dictionary using the functions, now including the placeholders
page_options = {
    "Text Generation": page_text_generation,
    "Text Summarization": page_summarization,
    "Sentiment Analysis": page_sentiment_analysis,
    "Named Entity Recognition (NER)": page_ner,
    "Question Answering (QA)": page_qa,
    "English to French Translation": page_translation,
    "Text Refinement (Paraphrase/Grammar)": page_paraphrase_grammar,
    "Semantic Text Similarity": page_text_similarity,
}

selection = st.sidebar.radio("Go to:", list(page_options.keys()))
page_options[selection]() # This line will now execute the placeholder functions

# Preload all models
with st.spinner("Preparing all AI models..."):
    load_generator()
    load_summarizer()
    load_sentiment_model()
    load_ner_model()
    load_qa_model()
    load_translator()
    load_paraphraser()
    load_grammar_corrector()
    load_similarity_model()
    time.sleep(0.5)

st.toast("All AI Models are ready!", icon='✅')
