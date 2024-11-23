# Import required libraries
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import fitz  # PyMuPDF
from prompt import PROMPT  # Import the prompt template


# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text("text")
    return text


# Response Format For Language Translation
def translation_chain(input_text, languages):

    # Define the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-002", temperature=1, api_key=GOOGLE_API_KEY
    )
    
    # Define the prompt
    prompt = PromptTemplate(
        input_variables=["text", "languages"],
        template=PROMPT,
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate response
    response = llm_chain.run({"text": input_text, "languages": languages})
    return response


# Streamlit app
st.set_page_config(page_title="Language Translator")
st.header("Language Translator")

# PDF file upload
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Input language specification
languages = st.text_input("Enter the language pair (e.g., 'English to Spanish')")

# Translate button
if st.button("Translate"):
    if pdf_file and languages:
        pdf_text = extract_text_from_pdf(pdf_file)
        response = translation_chain(input_text=pdf_text, languages=languages)
        st.write("The Translation is: \n\n", response)
    else:
        st.warning("Please enter both the text to translate and the language pair.")