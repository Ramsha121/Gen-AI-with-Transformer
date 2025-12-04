import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import time

# --- Configuration and Initialization ---

# Set Streamlit page configuration for a modern, wide layout
st.set_page_config(
    page_title="OGGen AI Transformer Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply a cohesive, theme-based style using a custom dark mode feel
st.markdown(
    """
    <style>
    /* Main App Styling */
    .stApp {
        background-color: #0d1117; /* Dark background similar to GitHub dark theme */
        color: #c9d1d9; /* Light text color */
    }
    /* Header and Title Styles */
    .stTitle, .stHeader {
        color: #58a6ff; /* A bright blue accent color */
    }
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc { /* Targeting sidebar containers */
        background-color: #161b22; /* Slightly lighter dark shade for sidebar */
    }
    /* Buttons */
    .stButton>button {
        background-color: #238636; /* GitHub green for actions */
        color: white;
        border-radius: 6px;
        border: 1px solid #30363d;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    /* Text Areas and Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
    }
    /* Success/Info Boxes */
    .stAlert {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Caching AI Models for Performance ---

@st.cache_resource(show_spinner="Initializing Text Generation Model (GPT-2)...")
def load_generator():
    """Load the text generation pipeline."""
    return pipeline("text-generation", model="gpt2")

@st.cache_resource(show_spinner="Initializing Summarization Model (BART)...")
def load_summarizer():
    """Load the summarization pipeline."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner="Initializing Sentiment Analysis Model...")
def load_sentiment_model():
    """Load the sentiment analysis pipeline."""
    return pipeline("sentiment-analysis")

@st.cache_resource(show_spinner="Initializing NER Model (BERT)...")
def load_ner_model():
    """Load the Named Entity Recognition pipeline."""
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource(show_spinner="Initializing Question Answering Model...")
def load_qa_model():
    """Load the Question Answering pipeline."""
    return pipeline("question-answering")

@st.cache_resource(show_spinner="Initializing Translation Model (EN-FR)...")
def load_translator():
    """Load the English-to-French translation pipeline."""
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

@st.cache_resource(show_spinner="Initializing Paraphrase Model (T5)...")
def load_paraphraser():
    """Load the text2text generation pipeline for paraphrasing."""
    return pipeline("text2text-generation", model="t5-small")

@st.cache_resource(show_spinner="Initializing Grammar Correction Model...")
def load_grammar_corrector():
    """Load the grammar correction pipeline."""
    return pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

@st.cache_resource(show_spinner="Initializing Similarity Encoder (MiniLM)...")
def load_similarity_model():
    """Load the Sentence Transformer model for semantic similarity."""
    # Using 'all-MiniLM-L6-v2' which is highly efficient
    return SentenceTransformer("all-MiniLM-L6-v2")


# --- Utility Functions (Logic remains the same as original script) ---

def run_text_generation(prompt, max_len):
    """Executes the text generation logic."""
    generator = load_generator()
    result = generator(prompt, max_length=max_len)
    return result[0]["generated_text"]

def run_summarization(text, max_len, min_len):
    """Executes the summarization logic."""
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]["summary_text"]

def run_sentiment_analysis(text):
    """Executes the sentiment analysis logic."""
    sentiment_model = load_sentiment_model()
    result = sentiment_model(text)
    return result[0]

def run_ner(text):
    """Executes the Named Entity Recognition logic."""
    ner = load_ner_model()
    return ner(text)

def run_qa(question, context):
    """Executes the Question Answering logic."""
    qa = load_qa_model()
    return qa(question=question, context=context)

def run_translation(text):
    """Executes the English-to-French translation logic."""
    translate = load_translator()
    return translate(text)[0]["translation_text"]

def run_paraphrasing(text):
    """Executes the paraphrasing logic."""
    para = load_paraphraser()
    # T5 model requires a "paraphrase:" prefix for this task
    return para(f"paraphrase: {text}")[0]["generated_text"]

def run_grammar_correction(text):
    """Executes the grammar correction logic."""
    gc = load_grammar_corrector()
    return gc(text)[0]["generated_text"]

def run_text_similarity(text_a, text_b):
    """Executes the text similarity logic using cosine similarity."""
    model = load_similarity_model()
    a = model.encode(text_a, convert_to_tensor=True)
    b = model.encode(text_b, convert_to_tensor=True)
    # Cosine similarity is a number between -1 (opposite) and 1 (identical)
    similarity = util.pytorch_cos_sim(a, b).item()
    return similarity


# --- Streamlit UI Pages ---

def ai_hub_header(title, icon):
    """Custom header component."""
    st.markdown(f'<h1 class="stTitle">{icon} {title}</h1>', unsafe_allow_html=True)
    st.divider()

def page_text_generation():
    ai_hub_header("Creative Text Generation", "✍️")

    with st.container(border=True):
        prompt = st.text_input("Enter a prompt to start the story/text:", value="The future of AI in healthcare is")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            max_len = st.slider("Max Output Length", 50, 500, 150)
        
        if st.button("Generate Text", key="gen_btn"):
            if prompt:
                with st.spinner(f"Generating up to {max_len} tokens..."):
                    generated_text = run_text_generation(prompt, max_len)
                    st.success("Generation Complete!")
                    st.markdown("---")
                    st.subheader("Generated Content")
                    st.code(generated_text, language="text")
            else:
                st.error("Please enter a prompt.")

def page_summarization():
    ai_hub_header("Document Summarization", "📚")
    
    default_text = """
    Artificial intelligence is transforming industries worldwide. Companies use AI to automate processes, 
    gain insights from data, and improve decision-making. As AI continues to advance, 
    its impact on society will grow significantly, leading to new jobs focused on AI development, 
    maintenance, and ethical oversight. The transition, however, requires careful planning and education.
    """
    
    text = st.text_area("Paste Text to Summarize:", default_text, height=200)

    col1, col2 = st.columns(2)
    with col1:
        min_len = st.slider("Minimum Summary Length", 10, 50, 10)
    with col2:
        max_len = st.slider("Maximum Summary Length", min_len, 200, 50)

    if st.button("Summarize", key="sum_btn"):
        if text:
            with st.spinner("Creating concise summary..."):
                summary = run_summarization(text, max_len, min_len)
                st.success("Summarization Complete!")
                st.markdown("---")
                st.subheader("Summary")
                st.info(summary)
        else:
            st.error("Please provide text to summarize.")

def page_sentiment_analysis():
    ai_hub_header("Sentiment Analysis", "😊/😠")
    
    text = st.text_input("Enter a sentence or phrase for analysis:", value="I love using machine learning tools—they make life easier!")
    
    if st.button("Analyze Sentiment", key="sent_btn"):
        if text:
            with st.spinner("Analyzing sentiment..."):
                result = run_sentiment_analysis(text)
                st.success("Analysis Complete!")
                st.markdown("---")
                st.subheader("Results")
                
                # Determine color and icon based on label
                label = result['label']
                score = result['score']
                
                if label == 'POSITIVE':
                    color = "#238636" # Green
                    icon = "⭐"
                elif label == 'NEGATIVE':
                    color = "#cf222e" # Red
                    icon = "🚨"
                else:
                    color = "#58a6ff" # Blue (for neutral/other, though model is usually binary)
                    icon = "❓"

                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: {color}20; border: 1px solid {color};">
                    <h3 style="color: {color}; margin: 0; display: flex; align-items: center;">
                        {icon} Sentiment: {label}
                    </h3>
                    <p style="margin: 5px 0 0 0;">Confidence Score: <strong>{score:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Please enter text for sentiment analysis.")

def page_ner():
    ai_hub_header("Named Entity Recognition (NER)", "👤📍")
    
    text = st.text_area("Paste text to find entities (People, Locations, Organizations):", 
                        value="Elon Musk founded SpaceX in California, and Amazon is investing $10 billion in India.", 
                        height=150)
    
    if st.button("Extract Entities", key="ner_btn"):
        if text:
            with st.spinner("Extracting entities..."):
                entities = run_ner(text)
                st.success("Extraction Complete!")
                st.markdown("---")
                st.subheader("Extracted Entities (Keywords)")
                
                # Prepare data for a visually appealing table
                data = [{"Entity": entity['word'], "Type": entity['entity_group'], "Score": f"{entity['score']:.4f}"} 
                        for entity in entities]
                
                if data:
                    df = pd.DataFrame(data)
                    # Use Streamlit's data_editor for a nice interactive display
                    st.data_editor(
                        df, 
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Entity": st.column_config.TextColumn("Entity Text", help="The extracted entity name"),
                            "Type": st.column_config.TextColumn("Entity Type", help="PERSON (PER), ORGANIZATION (ORG), LOCATION (LOC), etc."),
                            "Score": st.column_config.ProgressColumn("Confidence Score", help="Model's confidence (0.0 to 1.0)", format="%.4f", min_value=0.0, max_value=1.0)
                        }
                    )
                else:
                    st.info("No named entities were found in the text.")
        else:
            st.error("Please enter text for NER.")

def page_qa():
    ai_hub_header("Question Answering", "❓💡")
    
    st.markdown("Provide a source text (context) and ask a question about it. The AI will pinpoint the answer.")

    context = st.text_area("Context (Source Text):", 
                           value="The Taj Mahal is located in Agra, India. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal.", 
                           height=150)
    question = st.text_input("Question:", value="Where is Taj Mahal located?")
    
    if st.button("Get Answer", key="qa_btn"):
        if question and context:
            with st.spinner("Searching for the answer..."):
                result = run_qa(question, context)
                st.success("Answer Found!")
                st.markdown("---")
                
                answer = result['answer']
                score = result['score']

                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: #58a6ff1a; border: 1px solid #58a6ff;">
                    <h3 style="color: #58a6ff; margin: 0;">Answer: {answer}</h3>
                    <p style="margin: 5px 0 0 0;">Confidence: <strong>{score:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Source Context with Highlight")
                # Highlight the answer in the context
                start = result['start']
                end = result['end']
                
                highlighted_context = (
                    context[:start] + 
                    f"**<span style='background-color: yellow; color: black; padding: 2px 4px; border-radius: 4px;'>{context[start:end]}</span>**" + 
                    context[end:]
                )
                st.markdown(highlighted_context)
        else:
            st.error("Please provide both a context and a question.")

def page_translation():
    ai_hub_header("English to French Translation", "🌍🇫🇷")

    text = st.text_area("Enter English text to translate:", 
                        value="How are you? I hope this application helps you explore the power of natural language processing.", 
                        height=150)
    
    if st.button("Translate to French", key="trans_btn"):
        if text:
            with st.spinner("Translating..."):
                translation = run_translation(text)
                st.success("Translation Complete!")
                st.markdown("---")
                st.subheader("French Translation")
                st.info(translation)
        else:
            st.error("Please enter text to translate.")

def page_paraphrase_grammar():
    ai_hub_header("Text Refinement (Paraphrase & Grammar)", "📝✨")
    
    col1, col2 = st.columns(2)

    # --- Paraphrasing Section ---
    with col1:
        st.subheader("Paraphrase Generator")
        para_text = st.text_area("Enter text to rephrase:", 
                                 value="Machine learning is interesting, and it has many useful applications.", 
                                 height=100, key="para_in")
        
        if st.button("Paraphrase", key="para_btn"):
            if para_text:
                with st.spinner("Generating a new sentence structure..."):
                    paraphrased_text = run_paraphrasing(para_text)
                    st.success("Paraphrasing Complete!")
                    st.markdown("#### Result")
                    st.code(paraphrased_text, language="text")
            else:
                st.error("Please enter text to paraphrase.")

    # --- Grammar Correction Section ---
    with col2:
        st.subheader("Grammar & Spelling Corrector")
        gc_text = st.text_area("Enter text with errors:", 
                               value="She go to school every days but they is not happy.", 
                               height=100, key="gc_in")
        
        if st.button("Correct Grammar", key="gc_btn"):
            if gc_text:
                with st.spinner("Checking and correcting grammar..."):
                    corrected_text = run_grammar_correction(gc_text)
                    st.success("Correction Complete!")
                    st.markdown("#### Result")
                    st.code(corrected_text, language="text")
            else:
                st.error("Please enter text to correct.")


def page_text_similarity():
    ai_hub_header("Semantic Text Similarity", "🔗")
    
    st.markdown("This tool calculates how **semantically similar** two sentences are, using cosine similarity (a score from 0.0 to 1.0).")

    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_area("Sentence A:", value="AI will change the world.", height=100)
    with col2:
        text_b = st.text_area("Sentence B:", value="Artificial intelligence will transform industries.", height=100)
        
    if st.button("Calculate Similarity", key="sim_btn"):
        if text_a and text_b:
            with st.spinner("Calculating embedding vectors..."):
                similarity = run_text_similarity(text_a, text_b)
                st.success("Calculation Complete!")
                
                # Visual feedback on similarity score
                similarity_percent = similarity * 100
                
                # Determine color based on score (Green for high, Orange for medium, Red for low)
                if similarity >= 0.7:
                    color = "#238636"
                elif similarity >= 0.4:
                    color = "#fb8500"
                else:
                    color = "#cf222e"
                    
                st.markdown("---")
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 8px; text-align: center; background-color: {color}20; border: 2px solid {color};">
                    <h2 style="color: {color}; margin-bottom: 5px;">Similarity Score</h2>
                    <h1 style="color: {color}; font-size: 3em; margin-top: 0;">{similarity:.4f}</h1>
                    <p style="margin: 0;">({similarity_percent:.2f}% Match)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Optional: Show the underlying value for reference
                # st.metric(label="Cosine Similarity Score", value=f"{similarity:.4f}")
        else:
            st.error("Please enter both Sentence A and Sentence B.")


# --- Main Application Logic ---

# Use Streamlit's sidebar for navigation
st.sidebar.markdown("# **OGGen AI Hub**")
st.sidebar.markdown("Explore various NLP tasks powered by Hugging Face Transformers and Sentence Transformers.")

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

# Call the selected page function
page_options[selection]()

st.sidebar.markdown("---")
st.sidebar.caption("Logic is preserved from the original Python script. Models are cached for fast performance.")

# --- Pre-load all models in background to avoid long wait on first interaction ---
# This initiates the loading process when the script first runs.
with st.spinner("Preparing all AI models for a smooth experience..."):
    load_generator()
    load_summarizer()
    load_sentiment_model()
    load_ner_model()
    load_qa_model()
    load_translator()
    load_paraphraser()
    load_grammar_corrector()
    load_similarity_model()
    # Adding a small sleep to ensure the spinner is visible for a moment
    time.sleep(0.5)

st.toast("All AI Models are ready!", icon='✅')
