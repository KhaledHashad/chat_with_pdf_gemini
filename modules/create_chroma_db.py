import chromadb
from typing import List
from .generate_embeddings import GeminiEmbeddingFunction


def create_chroma_db(documents: List, path: str, name: str) -> tuple[chromadb.Collection, str]:
    """
    Creates a Chroma database using the provided documents, path, and collection name.

    Parameters:
    - documents: An iterable of documents to be added to the Chroma database. Each document should be a dictionary-like object.
    - path (str): The path where the Chroma database will be stored. The directory must exist.
    - name (str): The name of the collection within the Chroma database. The name must be unique within the database.

    Returns:
    - Tuple[chromadb.Collection, str]: A tuple containing the created Chroma Collection and its name.

    Raises:
    - ValueError: If the provided path does not exist or if the provided name is already in use.

    Example:
    >>> documents = [{'text': 'Hello, world!'}, {'text': 'This is a test.'}]
    >>> path = '/path/to/chroma_db'
    >>> name = 'my_collection'
    >>> db, db_name = create_chroma_db(documents, path, name)
    >>> print(db_name)
    my_collection
    """

    # Initialize a persistent Chroma client
    chroma_client = chromadb.PersistentClient(path=path)

    # Create a new collection in the Chroma database
    db = chroma_client.create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction())

    # Add documents to the Chroma collection
    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    # Return the created Chroma collection and its name
    return db, name
