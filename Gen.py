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
#   ADDING MISSING MODEL LOADERS (THIS FIXES THE NameError)
# =====================================================================

@st.cache_resource(show_spinner="Loading Paraphrasing Model...")
def load_paraphraser():
    return pipeline(
        "text2text-generation",
        model="Vamsi/T5_Paraphrase_Paws"
    )

@st.cache_resource(show_spinner="Loading Grammar Corrector...")
def load_grammar_corrector():
    return pipeline(
        "text2text-generation",
        model="prithivida/grammar_error_correcter_v1"
    )

@st.cache_resource(show_spinner="Loading Similarity Model...")
def load_similarity_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =====================================================================
#   EXISTING MODEL LOADERS
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
#   UI FUNCTIONS (unchanged – your layout preserved)
# =====================================================================

# (All your UI page functions stay exactly the same)
# I am not rewriting them to save space.
# Paste your entire original block of UI pages here without modifying.

# =====================================================================
#   NAVIGATION + MODEL PRELOAD
# =====================================================================

st.sidebar.markdown("# **OGGen AI Hub**")
st.sidebar.markdown("Explore various NLP tasks powered by Hugging Face Transformers.")

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
page_options[selection]()

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
