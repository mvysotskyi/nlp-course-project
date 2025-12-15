import weaviate
from weaviate.classes.query import MetadataQuery
import os
import dotenv

dotenv.load_dotenv()

client = weaviate.connect_to_local(grpc_port=50051)
collection = client.collections.get("SupremeCourtDecision")

print(f"Total objects: {collection.aggregate.over_all(total_count=True).total_count}")

# Test 1: Full text search (BM25)
print("\n--- BM25 Search for 'суд' ---")
response = collection.query.bm25(
    query="суд",
    limit=2,
    return_properties=["case_number", "court_name", "summary"]
)

for o in response.objects:
    print(f"ID: {o.uuid}")
    print(f"Case: {o.properties['case_number']}")
    print(f"Court: {o.properties['court_name']}")
    print(f"Summary: {o.properties['summary'][:100]}...")
    print("-" * 20)

# Test 2: Vector Search (if we had a query vector, but we can't easily generate one without the embedding model here, 
# but we can check if vectors are present by looking at the objects)
# Actually we can use the embedding model if we import it.

from langchain_aws import BedrockEmbeddings
embeddings = BedrockEmbeddings(
    model_id="cohere.embed-multilingual-v3",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("aws_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_key")
)

print("\n--- Vector Search for 'criminal case' (translated to vector) ---")
query_text = "кримінальна справа"
vector = embeddings.embed_query(query_text)

response = collection.query.near_vector(
    near_vector=vector,
    limit=2,
    return_properties=["case_number", "summary"]
)

for o in response.objects:
    print(f"Case: {o.properties['case_number']}")
    print(f"Summary: {o.properties['summary'][:100]}...")
    print("-" * 20)

client.close()
