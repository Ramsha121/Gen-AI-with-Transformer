import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------
# 🌈 PAGE CONFIG & STYLING
# ------------------------------------------
st.set_page_config(
    page_title="Interactive NLP App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .title {
            font-size:40px !important;
            color:#7b001c;
            font-weight:900;
            text-align:center;
        }
        .subtitle {
            font-size:20px !important;
            color:#3c3c3c;
            text-align:center;
            margin-bottom:20px;
        }
        .section-title {
            font-size:26px;
            color:#7b001c;
            font-weight:700;
        }
        .stTextInput > label, .stTextArea > label {
            color:#7b001c !important;
        }
        .block {
            padding:20px;
            border-radius:15px;
            background:#f6f0e8;
            margin-bottom:20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# 🎉 HEADER
# ------------------------------------------
st.markdown('<div class="title">⚡ Intelligent NLP Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore text generation, summarization, sentiment analysis, NER, translation & more — all powered by Transformers!</div>', unsafe_allow_html=True)
st.write("")

# -------------------------------------------------
# 🧠 LOAD YOUR TRANSFORMER MODELS
# -------------------------------------------------
generator = pipeline("text-generation", model="gpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_model = pipeline("sentiment-analysis")
ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering")
translate_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
para_model = pipeline("text2text-generation", model="t5-small")
grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# 🎨 SIDEBAR MENU
# -------------------------------------------------
menu = st.sidebar.radio(
    "Choose a Task",
    [
        "Text Generation",
        "Summarization",
        "Sentiment Analysis",
        "Named Entity Recognition (NER)",
        "Question Answering",
        "Translation",
        "Paraphrasing",
        "Grammar Correction",
        "Text Similarity"
    ]
)

# -------------------------------------------------
# 🧩 TEXT GENERATION
# -------------------------------------------------
if menu == "Text Generation":
    st.markdown('<div class="section-title">📝 Text Generation</div>', unsafe_allow_html=True)
    prompt = st.text_input("Enter a starting prompt:", "BJP is")

    if st.button("Generate"):
        with st.spinner("Generating..."):
            result = generator(prompt, max_length=400)
            st.success("Generated Text:")
            st.write(result[0]["generated_text"])

# -------------------------------------------------
# 🧩 SUMMARIZATION
# -------------------------------------------------
elif menu == "Summarization":
    st.markdown('<div class="section-title">📚 Summarization</div>', unsafe_allow_html=True)
    text = st.text_area("Enter text to summarize:", """Artificial intelligence is transforming industries worldwide...""")

    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
            st.success("Summary:")
            st.write(summary[0]["summary_text"])

# -------------------------------------------------
# 🧩 SENTIMENT ANALYSIS
# -------------------------------------------------
elif menu == "Sentiment Analysis":
    st.markdown('<div class="section-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    text = st.text_input("Enter text:", "I love using machine learning tools—they make life easier!")

    if st.button("Analyze"):
        result = sentiment_model(text)
        st.write(result)

# -------------------------------------------------
# 🧩 NER
# -------------------------------------------------
elif menu == "Named Entity Recognition (NER)":
    st.markdown('<div class="section-title">🔍 Named Entity Recognition</div>', unsafe_allow_html=True)
    text = st.text_input("Enter text:", "Elon Musk founded SpaceX in California.")

    if st.button("Identify Entities"):
        st.write(ner_model(text))

# -------------------------------------------------
# 🧩 QUESTION ANSWERING
# -------------------------------------------------
elif menu == "Question Answering":
    st.markdown('<div class="section-title">❓ Question Answering</div>', unsafe_allow_html=True)

    question = st.text_input("Question:", "Where is Taj Mahal?")
    context = st.text_area("Context:", "The Taj Mahal is located in Agra, India.")

    if st.button("Get Answer"):
        st.write(qa_pipeline(question=question, context=context))

# -------------------------------------------------
# 🧩 TRANSLATION
# -------------------------------------------------
elif menu == "Translation":
    st.markdown('<div class="section-title">🌍 Translation (EN → FR)</div>', unsafe_allow_html=True)
    text = st.text_input("Enter English text:", "How are you?")

    if st.button("Translate"):
        translated = translate_model(text)
        st.success("French Translation:")
        st.write(translated[0]["translation_text"])

# -------------------------------------------------
# 🧩 PARAPHRASING
# -------------------------------------------------
elif menu == "Paraphrasing":
    st.markdown('<div class="section-title">♻️ Paraphrasing</div>', unsafe_allow_html=True)
    text = st.text_input("Enter text to paraphrase:", "Machine learning is interesting")

    if st.button("Paraphrase"):
        result = para_model("paraphrase: " + text)
        st.write(result[0]["generated_text"])

# -------------------------------------------------
# 🧩 GRAMMAR CORRECTION
# -------------------------------------------------
elif menu == "Grammar Correction":
    st.markdown('<div class="section-title">📝 Grammar Correction</div>', unsafe_allow_html=True)
    text = st.text_input("Enter sentence:", "She go to school every days")

    if st.button("Correct"):
        corrected = grammar_corrector(text)
        st.write(corrected[0]["generated_text"])

# -------------------------------------------------
# 🧩 TEXT SIMILARITY
# -------------------------------------------------
elif menu == "Text Similarity":
    st.markdown('<div class="section-title">🔗 Text Similarity</div>', unsafe_allow_html=True)
    text1 = st.text_input("Text A:", "AI will change the world")
    text2 = st.text_input("Text B:", "Artificial intelligence will transform industries")

    if st.button("Compare"):
        a = similarity_model.encode(text1, convert_to_tensor=True)
        b = similarity_model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(a, b)
        st.write(similarity)

