import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")

# Mapping languages to mBART language codes
LANG_CODE_MAP = {
    'en': 'en_XX',
    'ta': 'ta_IN',
    'hi': 'hi_IN',
    'fr': 'fr_XX'
}

@st.cache_resource
def text_summary(text, keywords=None, src_lang='en', tgt_lang='en'):
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Clean the input text
    text = clean_text(text)

    if len(text) < 10:  # Ensure text is long enough
        return "Input text is too short for summarization."

    if keywords:
        sentences = text.split(". ")
        emphasized_text = ""
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                emphasized_text += sentence + ". " + sentence + ". "
            else:
                emphasized_text += sentence + ". "
        text = emphasized_text.strip()

    # Convert src_lang and tgt_lang to mBART language codes
    try:
        tokenizer.src_lang = LANG_CODE_MAP[src_lang]
    except KeyError:
        return f"Unsupported source language: {src_lang}. Please select a valid language."

    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True, padding=True)

    # Set the correct target language code
    try:
        forced_bos_token_id = tokenizer.lang_code_to_id[LANG_CODE_MAP[tgt_lang]]
    except KeyError:
        return f"Unsupported target language: {tgt_lang}. Please select a valid language."

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=200, early_stopping=True, forced_bos_token_id=forced_bos_token_id)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def extract_keywords(text, n_keywords=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([text])
    feature_names = np.array(tfidf.get_feature_names_out())
    sorted_indices = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    keywords = feature_names[sorted_indices[:n_keywords]]
    return keywords

src_lang = st.selectbox("Select input language", ["en", "ta", "hi", "fr"])
tgt_lang = st.selectbox("Select output language (Summary)", ["en", "ta", "hi", "fr"])

choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text using MBart Multilingual Model")
    input_text = st.text_area("Enter your text here")
    user_keywords = st.text_input("Enter keywords to focus on (comma-separated)")

    if input_text:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("*Your Input Text*")
                st.info(input_text)
            with col2:
                st.markdown("*Summary Result*")
                keywords = [kw.strip() for kw in user_keywords.split(",")] if user_keywords else None
                result = text_summary(input_text, keywords=keywords, src_lang=src_lang, tgt_lang=tgt_lang)
                st.success(result)

                extracted_keywords = extract_keywords(input_text)
                st.markdown("*Extracted Keywords*")
                st.info(", ".join(extracted_keywords))

elif choice == "Summarize Document":
    st.subheader("Summarize Document using MBart Multilingual Model")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    user_keywords = st.text_input("Enter keywords to focus on (comma-separated)")

    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")  # Make sure this function is defined elsewhere
                st.markdown("*Extracted Text is Below:*")
                st.info(extracted_text)
            with col2:
                st.markdown("*Summary Result*")
                keywords = [kw.strip() for kw in user_keywords.split(",")] if user_keywords else None
                doc_summary = text_summary(extracted_text, keywords=keywords, src_lang=src_lang, tgt_lang=tgt_lang)
                st.success(doc_summary)

                extracted_keywords = extract_keywords(extracted_text)
                st.markdown("*Extracted Keywords*")
                st.info(", ".join(extracted_keywords))