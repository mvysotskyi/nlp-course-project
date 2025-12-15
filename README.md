# NLP Course Project — Legal Assistant (RAG + OCR + Anonymization)

Streamlit app that can ingest legal documents (text/PDF/images), run OCR, optionally anonymize entities (NER), and answer questions using a Retrieval-Augmented Generation (RAG) pipeline backed by Weaviate and AWS Bedrock.

## Installation / Setup

Requirements: Python 3.12 and Docker for running Weaviate.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd services/rag-pipeline/
pip install -r requirements.txt
```

Environment variables used by the project (create a `.env` if you want):
- `BEDROCK_KEY` (required to run `app.py`)
- `LLM_BASE_URL` (optional; defaults to AWS Bedrock OpenAI-compatible endpoint)
- `aws_key_id`, `aws_secret_key` (required for the RAG pipeline embeddings/LLM in `services/rag-pipeline/`, you need to enter them in .env in services/rag-pipeline or export)
- `UPLOAD_S3_BUCKET`, `DOWNLOAD_S3_BUCKET` (optional; defaults to `legal-ocr-input` / `legal-ocr-output`)

## How to Run

### RAG pipeline (Weaviate + data ingestion)
1) Start Weaviate:
```bash
docker compose -f services/rag-pipeline/docker-compose.yaml up -d
```
2) Populate RAG:
```bash
python populate_rag_aws.py
python populate_from_csv.py
```

### Main Streamlit UI
```bash
streamlit run app.py
```





## Authors / Contributors
- Severyn Shykula
- Nazar Tkhir
- Mykola Vysotskyi
