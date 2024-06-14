# Job Application Analyzer

## Description

This project demonstrates a Retrieval-Augmented Generation (RAG) system leveraging ChromaDB for document storage and retrieval, and Gemini AI for generating embeddings and answers. The goal is to efficiently answer user queries by retrieving relevant passages from a database and generating comprehensive, user-friendly responses.

### System Design
![Alt text](./images/5.png)


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




### Step 6: Run the gradio app

```bash
python app.py
```
Paste the following url in the browser 

```bash
http://127.0.0.1:7860
```


## Contribution
Contributions are welcome! If you find any issues or have suggestions for improvement, please create an issue or submit a pull request on the project's GitHub repository.

