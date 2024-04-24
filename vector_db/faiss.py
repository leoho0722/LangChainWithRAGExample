from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings


class FaissVectorDB():
    """建立 FAISS VectorDB"""

    def __init__(
        self,
        texts: str | list[Document],
        embedding_model: Embeddings,
        folder_path: str = "faiss_index"
    ) -> None:
        self.texts = texts
        self.embedding_model = embedding_model
        self.folder_path = folder_path

        if isinstance(self.texts, str):
            self.vector_store = FAISS.from_texts(
                [self.texts],
                embedding=self.embedding_model
            )
        elif isinstance(self.texts, list):
            self.vector_store = FAISS.from_documents(
                self.texts,
                embedding=self.embedding_model
            )

    def save_to_local(self):
        """將 FAISS VectorDB 儲存到本地"""

        self.vector_store.save_local(folder_path=self.folder_path)

    def load_from_local(self) -> FAISS:
        """載入本地的 FAISS VectorDB

        Returns:
            vector_store (FAISS): FAISS Vector DB
        """

        self.vector_store = FAISS.load_local(
            folder_path=self.folder_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        return self.vector_store
