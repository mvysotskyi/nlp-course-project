import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv
import os
import dotenv
import time
import re

dotenv.load_dotenv()

# --- Configuration ---
CSV_FILE = '2024-5K-supreme-court-decisions-analyzed.csv'
CLASS_NAME = "SupremeCourtDecisionFullText"
BATCH_SIZE = 50  # This will be number of chunks now
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- 1. Connect to Local Weaviate ---
try:
    client = weaviate.connect_to_local(grpc_port=50051)
    print("Connected to Weaviate.")
except Exception as e:
    print(f"Error connecting to Weaviate: {e}")
    exit(1)

# --- 2. Define Weaviate Schema ---
if client.collections.exists(CLASS_NAME):
    print(f"Class '{CLASS_NAME}' already exists. Deleting to repopulate...")
    client.collections.delete(CLASS_NAME)

print(f"Creating class '{CLASS_NAME}'...")
client.collections.create(
    name=CLASS_NAME,
    vectorizer_config=Configure.Vectorizer.none(), # We provide vectors manually
    properties=[
        Property(name="text", data_type=DataType.TEXT, tokenization=Tokenization.WORD), # The chunk text
        Property(name="summary", data_type=DataType.TEXT), # Keep summary if available
        Property(name="analysis", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
        Property(name="case_number", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
        Property(name="date", data_type=DataType.TEXT),
        Property(name="court_name", data_type=DataType.TEXT),
        Property(name="article_codes_and_documents", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
        Property(name="original_id", data_type=DataType.TEXT), # Store original CSV ID
        Property(name="chunk_index", data_type=DataType.INT) # To know the order
    ]
)

# --- 3. Initialize Embedding Model ---
print("Loading Cohere embedding model on AWS Bedrock...")
embeddings = BedrockEmbeddings(
    model_id="cohere.embed-multilingual-v3",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("aws_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_key")
)
print("Embedding model loaded.")

# --- 4. Preprocessing and Splitting ---
def preprocess_text(text):
    if not text:
        return ""
    # Basic preprocessing
    text = text.strip()
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

# --- 5. Process CSV and Upload ---
collection = client.collections.get(CLASS_NAME)

def process_batch(batch_objs):
    if not batch_objs:
        return

    texts = [obj['text'] for obj in batch_objs]
    
    try:
        # Generate embeddings for the chunks
        vectors = embeddings.embed_documents(texts)
        
        with collection.batch.dynamic() as batch:
            for obj, vector in zip(batch_objs, vectors):
                properties = {
                    "text": obj.get('text', ''),
                    "summary": obj.get('summary', ''),
                    "analysis": obj.get('analysis', ''),
                    "case_number": obj.get('case_number', ''),
                    "date": obj.get('date', ''),
                    "court_name": obj.get('court_name', ''),
                    "article_codes_and_documents": obj.get('article_codes_and_documents', ''),
                    "original_id": obj.get('original_id', ''),
                    "chunk_index": obj.get('chunk_index', 0)
                }
                batch.add_object(
                    properties=properties,
                    vector=vector
                )
    except Exception as e:
        print(f"Error processing batch: {e}")

print(f"Reading {CSV_FILE}...")
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    current_batch = []
    count = 0
    
    for row in reader:
        original_text = row.get('text', '')
        if not original_text:
            continue
            
        # Preprocess
        clean_text = preprocess_text(original_text)
        
        # Split
        chunks = text_splitter.split_text(clean_text)
        
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "text": chunk,
                "summary": row.get('summary', ''),
                "analysis": row.get('analysis', ''),
                "case_number": row.get('case_number', ''),
                "date": row.get('date', ''),
                "court_name": row.get('court_name', ''),
                "article_codes_and_documents": row.get('article_codes_and_documents', ''),
                "original_id": row.get('id', ''),
                "chunk_index": i
            }
            current_batch.append(chunk_obj)
            
            if len(current_batch) >= BATCH_SIZE:
                process_batch(current_batch)
                count += len(current_batch)
                print(f"Processed {count} chunks...")
                current_batch = []
                # time.sleep(0.1) # Optional

    # Process remaining
    if current_batch:
        process_batch(current_batch)
        count += len(current_batch)
        print(f"Processed {count} chunks.")

print("Done.")
client.close()
