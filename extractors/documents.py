import os
import pathlib

from extractors.base import BaseExtractor
from extractors.pdf import PDFExtractor


class DocumentsExtractor(BaseExtractor):
    """Documents Extractor 用來從資料夾中提取所有文件內容"""

    def __init__(self):
        self.documents = []

    def extract(self, dir_path: str):
        """提取資料夾中所有文件內容"""

        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                self.extract()
            elif os.path.isfile(item_path):
                self._extract_document(item_path)
        return self.documents

    def _extract_document(self, file_path: str):
        """根據副檔名使用對應的 Extractor 來提取文件內容

        Args:
            file_path (str): 檔案路徑
        """

        if pathlib.Path(file_path).suffix == ".pdf":
            pdf_documents = PDFExtractor.extract(file_path)
            self.documents.extend(pdf_documents)
