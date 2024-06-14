import chromadb
from .generate_embeddings import GeminiEmbeddingFunction


def load_chroma_collection(path, name):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Args:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.

    Raises:
    - Exception: If the Chroma database or collection does not exist at the specified path.

    Note:
    - This function assumes that the Chroma database and the specified collection exist.
    - The GeminiEmbeddingFunction is used as the default embedding function for the collection.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(
        name=name, embedding_function=GeminiEmbeddingFunction())

    return db
