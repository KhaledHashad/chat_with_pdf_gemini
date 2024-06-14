import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.

    This class extends the EmbeddingFunction class and implements the __call__ method
    to generate embeddings for a given set of documents using the Gemini AI API.

    Parameters:
    - input (Documents): A collection of documents to be embedded.

    Returns:
    - Embeddings: Embeddings generated for the input documents.
    """

    def __call__(self, input: Documents) -> Embeddings:
        """
        Custom method to generate embeddings for a given set of documents using the Gemini AI API.

        This method overrides the __call__ method of the EmbeddingFunction class. It retrieves the
        Gemini API key from environment variables, configures the GenAI API with the key, and then
        generates embeddings for the input documents using the Gemini AI API.

        Parameters:
        - input (Documents): A collection of documents to be embedded. The Documents class is assumed to be
                            defined elsewhere in the codebase.

        Returns:
        - Embeddings: Embeddings generated for the input documents. The Embeddings class is assumed to be
                    defined elsewhere in the codebase.

        Raises:
        - ValueError: If the Gemini API key is not provided in the environment variables.
        """
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(
                "Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]
