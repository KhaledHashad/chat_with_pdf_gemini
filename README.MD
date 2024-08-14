# Chat with your PDF using Gemini

## Description

This project demonstrates a Retrieval-Augmented Generation (RAG) system leveraging ChromaDB for document storage and retrieval, and Gemini AI for generating embeddings and answers. The goal is to efficiently answer user queries by retrieving relevant passages from a database and generating comprehensive, user-friendly responses.

.

## Usage

### Step 1: Create a Virtual Python Environment

First, create a new virtual environment to ensure all dependencies are isolated from your main Python installation.

```bash
conda create -n chat_pdf python=3.9
```

### Step 2: Activate the Environment

```bash
conda activate chat_pdf
```

### Step 3: Install the Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Obtain Gemini-1.5-Pro API Key

To use this application, you need to obtain an API key for Gemini-1.5-Pro from [Google AI Studio](https://aistudio.google.com/app/). Once obtained, follow these steps:

1. Rename `.env.example` to `.env`.
2. Paste your API key as the value for `GEMINI_API_KEY` in the `.env` file.


### Step 5: Run the Streamlit app

```bash
streamlit run app.py
```
Paste the following url in the browser 

```bash
http://localhost:8501
```


## Contribution
Contributions are welcome! If you find any issues or have suggestions for improvement, please create an issue or submit a pull request on the project's GitHub repository.

