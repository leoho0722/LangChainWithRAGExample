from dotenv import load_dotenv
from langchain.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from vector_db.faiss import FaissVectorDB
from utils.device import available_device

load_dotenv(override=True)


TEMPLATE = """Answer the question based only on the following context:
{context}

Note: You must answer with Traditional Chinese in Taiwan, but don't tell me what the language is
Question: {question}
"""


class OllamaRAG:
    """RAG (Retrieval Augmented Generation) Instance via Ollama"""

    def __init__(
        self,
        model_name: str,
        texts: str | list[Document],
        template: str = TEMPLATE,
        embedding_model: Embeddings = None,
        top_k: int = 3
    ):
        self.texts = texts

        device = available_device()
        print(f"Device: {device}")

        # 載入 Embedding Model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={
                    "device": device,
                    "trust_remote_code": True,
                },
            )

        # 建立 FAISS VectorDB
        faiss_vector_db = FaissVectorDB(
            texts=self.texts,
            embedding_model=self.embedding_model
        )
        faiss_vector_db.save_to_local()
        self.vector_store = faiss_vector_db.load_from_local()
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k
            }
        )

        # 定義問答 Template
        self.template = template if template else TEMPLATE
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # 載入 Ollama 內的 LLM
        self.model = Ollama(model=model_name, num_gpu=1)

        # 設定 LangChain
        self.chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )
