from pypdf import PdfReader
import re


def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    # Logic to read pdf
    reader = PdfReader(file_path)

    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text


def split_text(text: str) -> list[str]:
    """
    Splits a text string into a list of non-empty substrings based on the specified pattern.
    The "\n \n" pattern will split the document paragraph by paragraph.

    Parameters:
    - text (str): The input text to be split. This should be a string containing the text to be split.

    Returns:
    - List[str]: A list containing non-empty substrings obtained by splitting the input text.
                 Each element in the list represents a paragraph from the original text.

    Raises:
    - None

    Examples:
    >>> split_text("This is a test.\n \nThis is another test.")
    ["This is a test.", "This is another test."]
    """
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]
