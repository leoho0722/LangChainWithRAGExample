from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 載入 PDF 文件
loader = PyMuPDFLoader("駕駛人手冊.pdf")
PDF_data = loader.load()

# 將 PDF 文件中的文字進行分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)

# 載入 Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {
    'device': 'cpu'
}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

# 將 Embedding 結果匯入 VectorDB
persist_directory = 'db'
vectordb = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# 使用 LangChain 的 LlamaCpp
model_path = "llama.cpp/models/llama-2-7b-chat/llama-2_q4.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
