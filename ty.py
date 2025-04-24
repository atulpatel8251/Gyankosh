import fitz  # PyMuPDF
import streamlit as st

def extract_text_from_pdf(pdf_file):
    # Open the PDF using the file object (not the path)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit interface
st.title("PDF Text Extraction")
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None:
    text = extract_text_from_pdf(uploaded_pdf)
    st.write("Extracted Text:", text)
