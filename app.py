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
    This function retrieves the most relevant passage from a given database based on a query.

    Parameters:
    query (str): The user's query or question.
    db (ChromaDB): The ChromaDB instance where the documents are stored.
    n_results (int): The number of top results to return.

    Returns:
    passage (str): The most relevant passage from the database.

    Raises:
    None

    Note:
    This function assumes that the ChromaDB instance is already initialized and populated with documents.
    """
    passage = db.query(query_texts=[query], n_results=n_results)[
        'documents'][0]
    print(len(db.query(query_texts=[query], n_results=n_results)[
        'documents']))
    return passage


def make_rag_prompt(query, relevant_passage):
    """
    This function generates a prompt for the RAG (Retrieval-Augmented Generation) model.
    The prompt includes a question and a relevant passage from a document.

    Parameters:
    query (str): The user's query or question.
    relevant_passage (str): The most relevant passage from a document.

    Returns:
    prompt (str): The generated prompt for the RAG model.

    Note:
    The prompt is formatted according to the specifications of the RAG model.
    It includes instructions, the question, and the relevant passage.
    """

    # Escape special characters in the relevant passage
    escaped = relevant_passage.replace(
        "'", "").replace('"', "").replace("\n", " ")
    #print(relevant_passage)
    # Generate the prompt
    prompt = (f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
    strike a friendly and conversational tone. 
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """)

    return prompt


def generate_answer(prompt):
    """
    This function generates an answer to a given prompt using the Gemini-Pro model from the GenerativeAI library.

    Parameters:
    prompt (str): The prompt to be used for generating the answer. The prompt should be a question that refers to a relevant passage from a document.

    Returns:
    str: The generated answer to the given prompt.

    Raises:
    ValueError: If the Gemini API Key is not provided as an environment variable.

    Note:
    This function assumes that the Gemini API Key is set as an environment variable named "GEMINI_API_KEY".
    It also assumes that the GenerativeAI library is installed and properly configured.
    """

    # Retrieve the Gemini API Key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Check if the Gemini API Key is provided
    if not gemini_api_key:
        raise ValueError(
            "Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")

    # Configure the GenerativeAI library with the Gemini API Key
    genai.configure(api_key=gemini_api_key)

    # Initialize the Gemini-Pro model
    model = genai.GenerativeModel('gemini-pro')

    # Generate the answer to the prompt using the Gemini-Pro model
    answer = model.generate_content(prompt)

    # Return the generated answer
    return answer.text


def get_answer(db, query):
    """
    This function retrieves the answer to a given query using the relevant passage from the database.
    It generates a prompt for the RAG (Retrieval-Augmented Generation) model,
    and then uses the Gemini-Pro model from the GenerativeAI library to generate the answer.

    Parameters:
    db (ChromaDB): The ChromaDB instance where the documents are stored.
    query (str): The user's query or question.

    Returns:
    str: The generated answer to the given query.

    Raises:
    ValueError: If the Gemini API Key is not provided as an environment variable.

    Note:
    This function assumes that the ChromaDB instance is already initialized and populated with documents.
    It also assumes that the Gemini API Key is set as an environment variable named "GEMINI_API_KEY".
    It uses the get_relevant_passage, make_rag_prompt, and generate_answer functions to accomplish this.
    """

    relevant_text = get_relevant_passage(query, db, n_results=1)
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = generate_answer(prompt)
    return answer


# Streamlit App
st.title("Chat with Your Documents")

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

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
            db = load_chroma_collection(path="./ChromaDB", name=name)

        st.success('PDF processed successfully!')

    # Query input
    query = st.text_input("Enter your question:", key="query")
    if query:
        with st.spinner('Generating answer...'):
            answer = get_answer(db, query)
            # Append the question and answer to the conversation history
            st.session_state.conversation.append((query, answer))

# Chat interface styling


def display_message(message, is_user_message=True):
    """
    This function displays a message in the Streamlit app with appropriate styling.

    Parameters:
    message (str): The message to be displayed.
    is_user_message (bool): A flag indicating whether the message is from the user (True) or the bot (False). Default is True.

    Returns:
    None

    Note:
    This function uses the Streamlit library to display the message with appropriate styling.
    The message is wrapped in a div with a background color and padding.
    If is_user_message is True, the background color is set to #DCF8C6, otherwise it is set to #F1F0F0.
    """

    if is_user_message:
        st.markdown(f"""
        <div style="background-color: #DCF8C6; padding: 10px; border-radius: 10px; max-width: 60%;">
            <p style="margin: 0; word-wrap: break-word;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #F1F0F0; padding: 10px; border-radius: 10px; max-width: 60%;">
            <p style="margin: 0; word-wrap: break-word;">{message}</p>
        </div>
        """, unsafe_allow_html=True)


# Display the conversation history
if 'conversation' in st.session_state:
    for question, answer in st.session_state.conversation:
        display_message(f"**Question:** {question}", is_user_message=True)
        display_message(f"**Answer:** {answer}", is_user_message=False)
