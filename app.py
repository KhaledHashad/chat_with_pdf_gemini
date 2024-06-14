from modules.load_db import load_chroma_collection
from modules.create_chroma_db import create_chroma_db
from modules.load_pdf import load_pdf, split_text
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


pdf_text = load_pdf(file_path="../examples/Khalil_sResumeAPR2024.pdf")
chunked_text = split_text(text=pdf_text)

db, name = create_chroma_db(documents=chunked_text,
                            path="../ChromaDB",
                            name="rag_experiment")

db = load_chroma_collection(
    path="./ChromaDB", name="rag_experiment")


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
    strike a friendly and converstional tone. \
    If the passage is irrelevant to the ansimport os
from dotenv import load_dotenvwer, you may ignore it.
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
    # retrieve top 3 relevant text chunks
    relevant_text = get_relevant_passage(query, db, n_results=3)
    prompt = make_rag_prompt(query,
                             relevant_passage="".join(relevant_text))  # joining the relevant chunks to create a single passage
    answer = generate_answer(prompt)

    return answer


answer = get_answer(
    db, query="List the name of companies he did internships at")
print(answer)
