from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

model_name = "BAAI/bge-large-en"
model_kwargs = {"device" :"cpu"}
encode_kwargs ={"normalize_embedding": False}

embedding = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

url = "http://localhost:6333"
collection_name = "ai_db"

client = QdrantClient(
    url = url,
    prefer_grpc = False
)

print(client)

db = Qdrant(
    client = client,
    embeddings = embedding,
    collection_name = collection_name
)

query =  "90% of the world's data has been generated in how many years?"

docs = db.similarity_search_with_score(query = query, k=5)

for i in docs:
    doc, score = i
    print({"score..................................": score, "content": doc.page_content, "metadata": doc.metadata})
