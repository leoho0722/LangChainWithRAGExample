import os
from typing_extensions import deprecated

from langchain_community.document_loaders.pdf import (
    PyMuPDFLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from extractors.base import BaseExtractor


class PDFExtractor(BaseExtractor):
    """PDF Extractor 用來從 PDF 檔案中提取文字內容"""

    @staticmethod
    def extract(path: str):
        """根據路徑提取 PDF 文件內容

        Args:
            path (str): 檔案/資料夾路徑

        Returns:
            List[Document]: 讀取到的所有 PDF 檔案內容
        """

        global loader
        global pdf_data

        try:
            if os.path.isdir(path):
                loader = PyPDFDirectoryLoader(path)
                pdf_data = loader.load()
            elif os.path.isfile(path):
                loader = PyMuPDFLoader(path)
                pdf_data = loader.load()

            text_spiltter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=5
            )
            return text_spiltter.split_documents(pdf_data)
        except Exception as e:
            print(f"Extracting from PDF failed: {e}")
            return None

    @staticmethod
    @deprecated("Use `extract` method instead")
    def extract_text_from_file(file_path: str):
        """使用 PyMuPDFLoader 讀取與載入 PDF 檔案

        Args:
            file_path (str): PDF 檔案路徑

        Returns:
            List[Document]: 讀取到的所有 PDF 檔案內容
        """

        try:
            loader = PyMuPDFLoader(file_path)
            pdf_data = loader.load()

            text_spiltter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=5
            )
            return text_spiltter.split_documents(pdf_data)
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None

    @staticmethod
    @deprecated("Use `extract` method instead")
    def extract_text_from_directory(dir_path: str):
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
        except Exception as e:
            print(f"Extracting text from PDF failed: {e}")
            return None
