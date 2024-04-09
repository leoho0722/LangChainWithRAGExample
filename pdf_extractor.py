from langchain_community.document_loaders.pdf import PyMuPDFLoader


class PDFExtractor:
    """PDF Extractor 用來從 PDF 檔案中提取文字內容"""

    @staticmethod
    def extract_text(filepath):
        try:
            # 使用 PyMuPDFLoader 讀取與載入 PDF 檔案
            loader = PyMuPDFLoader(filepath)
            return loader.load()
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None
