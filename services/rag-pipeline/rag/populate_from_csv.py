import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from langchain_aws import BedrockEmbeddings
import csv
import os
import dotenv
import time

dotenv.load_dotenv()

# --- Configuration ---
CSV_FILE = '../2024-5K-supreme-court-decisions-analyzed.csv'
CLASS_NAME = "SupremeCourtDecision"
BATCH_SIZE = 50  # Adjust based on Bedrock limits and memory

# --- 1. Connect to Local Weaviate ---
try:
    client = weaviate.connect_to_local(grpc_port=50051)
    print("Connected to Weaviate.")
except Exception as e:
    print(f"Error connecting to Weaviate: {e}")
    exit(1)

# --- 2. Define Weaviate Schema ---
# We want to embed 'summary' for vector search.
# We want 'text', 'analysis', 'case_number', etc. for full-text search (BM25).

if client.collections.exists(CLASS_NAME):
    print(f"Class '{CLASS_NAME}' already exists. Deleting to repopulate...")
    client.collections.delete(CLASS_NAME)

print(f"Creating class '{CLASS_NAME}'...")
client.collections.create(
    name=CLASS_NAME,
    vectorizer_config=Configure.Vectorizer.none(), # We provide vectors manually
    properties=[
        Property(name="summary", data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT, tokenization=Tokenization.WORD), # Enable full-text search
        Property(name="analysis", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
        Property(name="case_number", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
        Property(name="date", data_type=DataType.TEXT),
        Property(name="court_name", data_type=DataType.TEXT),
        Property(name="article_codes_and_documents", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
        Property(name="original_id", data_type=DataType.TEXT) # Store original CSV ID
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

# --- 4. Process CSV and Upload ---
collection = client.collections.get(CLASS_NAME)

def process_batch(batch_objs):
    if not batch_objs:
        return

    summaries = [obj['summary'] for obj in batch_objs]
    
    try:
        # Generate embeddings for the summaries
        vectors = embeddings.embed_documents(summaries)
        
        with collection.batch.dynamic() as batch:
            for obj, vector in zip(batch_objs, vectors):
                properties = {
                    "summary": obj.get('summary', ''),
                    "text": obj.get('text', ''),
                    "analysis": obj.get('analysis', ''),
                    "case_number": obj.get('case_number', ''),
                    "date": obj.get('date', ''),
                    "court_name": obj.get('court_name', ''),
                    "article_codes_and_documents": obj.get('article_codes_and_documents', ''),
                    "original_id": obj.get('id', '')
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
        # Skip rows with empty summary if necessary, or handle them
        if not row.get('summary'):
            continue
            
        current_batch.append(row)
        
        if len(current_batch) >= BATCH_SIZE:
            process_batch(current_batch)
            count += len(current_batch)
            print(f"Processed {count} records...")
            current_batch = []
            # Optional: sleep to avoid rate limits if needed
            # time.sleep(0.5) 

    # Process remaining
    if current_batch:
        process_batch(current_batch)
        count += len(current_batch)
        print(f"Processed {count} records.")

print("Done.")
client.close()
