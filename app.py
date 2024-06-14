import streamlit as st
from modules.load_db import load_chroma_collection
from modules.create_chroma_db import create_chroma_db
from modules.load_pdf import load_pdf, split_text
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




def get_relevant_passage(query, db, n_results):
    """
    This function retrieves the most relevant passage from a database based on a given query.

    Parameters:
    query (str): The question or query for which the relevant passage needs to be found.
    db (ChromaDB): The ChromaDB instance containing the documents.
    n_results (int): The number of top results to return.

    Returns:
    dict: A dictionary containing the relevant passage. The dictionary has the following structure:
        {
            'documents': [
                {
                    'text': str,  # The text of the relevant passage.
                    'score': float  # The relevance score of the passage.
                },
                ...
            ]
        }

    Note:
    This function assumes that the ChromaDB instance is already initialized and populated with documents.
    """
    passage = db.query(query_texts=[query], n_results=n_results)[
        'documents'][0]
    return passage


def make_rag_prompt(query, relevant_passage):
    """
    This function generates a prompt for the RAG (Retrieval-Augmented Generation) model.
    The prompt includes a question and a relevant passage from a document.

    Parameters:
    query (str): The question to be answered.
    relevant_passage (str): The relevant passage from a document.

    Returns:
    str: The generated prompt for the RAG model.

    Note:
    The prompt is formatted according to the requirements of the RAG model.
    It includes instructions, the question, and the relevant passage.
    """

    # Escape special characters in the relevant passage
    escaped = relevant_passage.replace(
        "'", "").replace('"', "").replace("\n", " ")

    # Generate the prompt
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
    """
    This function generates an answer to a given prompt using the Gemini-Pro model.

    Parameters:
    prompt (str): The prompt to be used for generating the answer. The prompt should be a string that includes the question and relevant passage.

    Returns:
    str: The generated answer to the prompt.

    Raises:
    ValueError: If the Gemini API Key is not provided. The API Key should be set as an environment variable named "GEMINI_API_KEY".

    Note:
    This function uses the 'gemini-pro' model from the generativeai library to generate the answer.
    """

    # Retrieve the Gemini API Key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Check if the Gemini API Key is provided
    if not gemini_api_key:
        raise ValueError(
            "Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")

    # Configure the generativeai library with the Gemini API Key
    genai.configure(api_key=gemini_api_key)

    # Create an instance of the 'gemini-pro' model
    model = genai.GenerativeModel('gemini-pro')

    # Generate an answer using the 'gemini-pro' model
    answer = model.generate_content(prompt)

    # Return the generated answer
    return answer.text


def get_answer(db, query):
    """
    This function generates an answer to a given query using a relevant passage from a database.

    Parameters:
    db (ChromaDB): The ChromaDB instance containing the documents.
    query (str): The question to be answered.

    Returns:
    str: The generated answer to the query.

    Raises:
    ValueError: If the Gemini API Key is not provided.
    """

    # Get the relevant passage from the database
    relevant_text = get_relevant_passage(query, db, n_results=3)

    # Join the relevant chunks to create a single passage
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))

    # Generate an answer using the RAG model
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
