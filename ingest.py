from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sentence_transformers

loader = PyPDFLoader("introduction-to-ai.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
texts = text_splitter.split_documents(documents)

model_name = "BAAI/bge-large-en"
model_kwargs = {"device" :"cpu"}
encode_kwargs ={"normalize_embedding": False}

embedding = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

print("embedding..........")

url = "http://localhost:6333"
collection_name = "ai_db"

qdrant = Qdrant.from_documents(
    texts,
    embedding,
    url = url,
    prefer_grpc = False,
    collection_name = collection_name
)

