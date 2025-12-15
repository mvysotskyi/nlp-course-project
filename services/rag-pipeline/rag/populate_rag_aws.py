import weaviate
from weaviate.classes.config import Configure, Property, DataType 

from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document 
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

import boto3
import dotenv
import os

dotenv.load_dotenv()

from pdf_reader import PDFReader

# --- 1. Connect to Local Weaviate ---
client = weaviate.connect_to_local(
    grpc_port=50051
)
client.is_ready()

# --- 2. Define Weaviate Schema (Class) ---
class_name = "CohereBedrockEmbed"

if not client.collections.exists(class_name):
    print(f"Creating class '{class_name}'...")
    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT) 
        ]
    )
else:
    print(f"Class '{class_name}' already exists.")


# --- 3. Initialize Embedding Model ---
print("Loading Cohere embedding model on AWS Bedrock...")
embeddings = BedrockEmbeddings(
    model_id="cohere.embed-multilingual-v3",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("aws_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_key")
)
print("Embedding model loaded.")


# --- 4. Read and Split PDF ---
pdf_reader = PDFReader()
document = pdf_reader.read_pdf("../docs/codecs_ukr.pdf")

# Split by "Стаття" to get individual articles
print("Splitting document by 'Стаття' (articles)...")
pattern = r'(Стаття\s+\d+[^\n]*(?:\n(?!Стаття\s+\d+)[^\n]*)*)'
matches = re.findall(pattern, document, re.MULTILINE)

if matches:
    documents_chunks = matches
    print(f"Found {len(matches)} articles using regex pattern")
else:
    # Fallback: simple split by "Стаття"
    parts = re.split(r'(?=Стаття)', document)
    documents_chunks = [part.strip() for part in parts if part.strip() and 'Стаття' in part]
    print(f"Found {len(documents_chunks)} articles using simple split")

print(f"Total chunks created: {len(documents_chunks)}")
if len(documents_chunks) > 0:
    print(f"Sample chunk (first article): {documents_chunks[0][:200]}...")

for i in range(len(documents_chunks)):
    if len(documents_chunks[i]) > 2048:
        documents_chunks[i] = documents_chunks[i][:2047]
print(max([len(c) for c in documents_chunks]))
# --- 5. Add Data (Client-Side Embeddings) ---
my_collection = client.collections.get(class_name)

if my_collection.aggregate.over_all(total_count=True).total_count == 0:
    print("Ingesting data...")
    documents = [Document(page_content=t) for t in documents_chunks]
    
    vector_store = WeaviateVectorStore.from_documents(
        client=client,
        index_name=class_name,
        documents=documents,
        embedding=embeddings,
        text_key="text"
    )
    print("Data ingestion complete.")
else:
    print("Data already exists, skipping ingestion.")

# Clean up
client.close()
print("Population complete. Vector store is ready for querying.")
