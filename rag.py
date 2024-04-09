from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents.base import Document
from dotenv import load_dotenv

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
        model_name: str = "gpt-3.5-turbo",
        template: str = TEMPLATE,
        embedding_model: Embeddings = None,  # type: ignore
        top_k: int = 10
    ):
        self.texts = texts
        self.embedding_model = embedding_model if embedding_model is not None else OpenAIEmbeddings()

        # 建立 FAISS VectorDB
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
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def get_answer(self, question):
        """取得 LLM 的回答"""

        return self.chain.invoke(question)
