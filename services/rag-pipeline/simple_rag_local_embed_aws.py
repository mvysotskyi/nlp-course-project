import weaviate
from weaviate.classes.config import Configure, Property, DataType 

# --- FIXED: Import the correct class name 'WeaviateVectorStore' ---
from langchain_weaviate import WeaviateVectorStore

from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document 

from langchain_aws import ChatBedrock, BedrockEmbeddings
import boto3
import dotenv
import os

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Connect to Local Weaviate ---
client = weaviate.connect_to_local(
    grpc_port=50051
)
client.is_ready()

# --- 2. Define Weaviate Schema (Class) ---
class_name = "MyDocumentBedrockEmbed"

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
print("Loading AWS Bedrock embedding model...")
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("aws_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_key")
)
print("Embedding model loaded.")




from pdf_reader import PDFReader

pdf_reader = PDFReader()
document = pdf_reader.read_pdf("docs/codecs_ukr.pdf")

overlap = 40
chunk = 200
documents_chunks = []
print(len(document))
for i in range(0, len(document)//40, chunk - overlap):
    documents_chunks.append(document[max(0, i - overlap):min(i + chunk + overlap, len(document))])
print(f"Total chunks created: {len(documents_chunks)}")
print(f"Sample chunk: {documents_chunks[5]}")
# --- 4. Add Data (Client-Side Embeddings) ---
my_collection = client.collections.get(class_name)

if my_collection.aggregate.over_all(total_count=True).total_count == 0:
    print("Ingesting data...")
    documents = [Document(page_content=t) for t in documents_chunks]
    
    vector_store = WeaviateVectorStore.from_documents(
        client=client,
        collection_name=class_name,
        documents=documents,
        embedding=embeddings,
        text_key="text"
    )
    print("Data ingestion complete.")
else:
    print("Data already exists, skipping ingestion.")
    vector_store = WeaviateVectorStore(
        client=client,
        collection_name=class_name,
        embedding=embeddings,
        text_key="text"
    )


# --- 5. Define the RAG Chain (Using AWS Bedrock) ---
aws_region = "us-east-1"
boto3.setup_default_session(region_name=aws_region)
import dotenv
import os
dotenv.load_dotenv()

llm = ChatBedrock(
    model_id="openai.gpt-oss-120b-1:0",
    model_kwargs={"temperature": 0.1},
    region_name=aws_region,

    aws_access_key_id=os.getenv("aws_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_key")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# --- 5. Define the RAG Chain (Using AWS Bedrock) ---
aws_region = "us-east-1"
boto3.setup_default_session(region_name=aws_region)

llm = ChatBedrock(
    model_id="openai.gpt-oss-120b-1:0",
    model_kwargs={"temperature": 0.1},
    region_name=aws_region,

    aws_access_key_id=os.getenv("aws_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_key")
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

template = """
You are an assistant for question-answering tasks.
Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs[:2]])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. Run the Chain ---
print("\n--- Asking RAG question (with AWS Bedrock) ---")
question = "Cкільки дають термін, якщо вкрасти щось з магазину?"
response = chain.invoke(question)

print(f"Question: {question}")
print(f"Answer: {response}")

# Clean up
client.close()