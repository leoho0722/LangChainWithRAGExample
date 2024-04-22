from langchain_community.document_loaders.pdf import PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFExtractor:
    """PDF Extractor 用來從 PDF 檔案中提取文字內容"""

    @staticmethod
    def extract_text(file_path):
        """使用 PyMuPDFLoader 讀取與載入 PDF 檔案

        Args:
            filepath (str): PDF 檔案路徑

        Returns:
            List[Document]: 讀取到的所有 PDF 檔案內容
        """

        try:
            loader = PyMuPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None

    @staticmethod
    def extract_text_from_directory(dir_path):
        """使用 PyPDFDirectoryLoader 讀取與載入資料夾內所有的 PDF 檔案

        Args:
            dir_path (str): 資料夾路徑

        Returns:
            List[Document]: 讀取到的所有 PDF 檔案內容
        """

        try:
            loader = PyPDFDirectoryLoader(dir_path)
            pdf_data = loader.load()

            text_spiltter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=5
            )
            return text_spiltter.split_documents(pdf_data)

            # return loader.load()
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None
