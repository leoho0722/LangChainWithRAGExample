import PyPDF2
from langchain_core.documents.base import Document


class PDFExtractor:
    """PDF Extractor 用來從 PDF 檔案中提取文字內容"""

    @staticmethod
    def extract_text(filepath):
        try:
            with open(filepath, "rb") as f:
                # 取得 PDF 檔案內容
                pdf_reader = PyPDF2.PdfReader(f)

                # 逐頁提取 PDF 檔案內容
                return [Document(page_content=page.extract_text()) for page in pdf_reader.pages]
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None
