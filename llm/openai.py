from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from vector_db.faiss import FaissVectorDB

load_dotenv(override=True)


TEMPLATE = """Answer the question based only on the following context:
{context}

Note: You must answer with Traditional Chinese in Taiwan, but don't tell me what the language is
Question: {question}
"""


class RAG():
    """RAG (Retrieval Augmented Generation) Instance"""

    def __init__(
        self,
        texts: str | list[Document],
        model_name: str = "gpt-4o",
        template: str = TEMPLATE,
        embedding_model: Embeddings = None,
        top_k: int = 3
    ):
        self.texts = texts
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = OpenAIEmbeddings()

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

        # 初始化 OpenAI Model
        self.model = ChatOpenAI(model=model_name)

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
