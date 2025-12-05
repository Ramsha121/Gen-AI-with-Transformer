import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------------------------------------
#  PARAPHRASER LOADER  (SAFE FOR STREAMLIT CLOUD)
# -------------------------------------------------------------
@st.cache_resource
def load_paraphraser():
    model_name = "ramsrigouthamg/t5_paraphraser"  # SAFE MODEL (no sentencepiece issues)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    paraphrase_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    return paraphrase_pipeline


# -------------------------------------------------------------
#  PARAPHRASER PAGE (CALL THIS IN YOUR MAIN PAGES DICT)
# -------------------------------------------------------------
def page_paraphraser():
    st.title("📝 Paraphrasing Tool")

    st.write("Enter any sentence or paragraph below and I will transform it into a fresh, reworded version while keeping the meaning intact.")

    text = st.text_area("Enter text to paraphrase:")

    if st.button("Paraphrase"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            paraphraser = load_paraphraser()
            output = paraphraser(text)[0]['generated_text']
            st.success(output)
            
