import PyPDF2
from langchain_core.documents.base import Document


class PDFExtractor:

    @staticmethod
    def extract_text(filepath):
        try:
            with open(filepath, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return [Document(page_content=page.extract_text()) for page in pdf_reader.pages]
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None
