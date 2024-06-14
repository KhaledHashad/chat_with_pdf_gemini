import streamlit as st
from modules.load_db import load_chroma_collection
from modules.create_chroma_db import create_chroma_db
from modules.load_pdf import load_pdf, split_text
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper functions


def get_relevant_passage(query, db, n_results):
    passage = db.query(query_texts=[query], n_results=n_results)[
        'documents'][0]
    return passage


def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace(
        "'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and conversational tone. \
If the passage is irrelevant to the answer, you may ignore it.
QUESTION: '{query}'
PASSAGE: '{relevant_passage}'

ANSWER:
""").format(query=query, relevant_passage=escaped)
    return prompt


def generate_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text


def get_answer(db, query):
    relevant_text = get_relevant_passage(query, db, n_results=3)
    # Joining the relevant chunks to create a single passage
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = generate_answer(prompt)
    return answer


# Streamlit App
st.title("Chat with Your Documents")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner('Processing the PDF...'):
        # Load and process the PDF
        pdf_text = load_pdf(file_path=uploaded_file)
        chunked_text = split_text(text=pdf_text)

        # Extract file name without extension for the collection name
        name = os.path.splitext(uploaded_file.name)[0]

        # Create or load Chroma DB collection
        try:
            db, name = create_chroma_db(
                documents=chunked_text, path="./ChromaDB", name=name)
        except:
            db = load_chroma_collection(
                path="./ChromaDB", name=name)

        st.success('PDF processed successfully!')

    # Query input
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner('Generating answer...'):
            answer = get_answer(db, query)
            st.write(answer)
