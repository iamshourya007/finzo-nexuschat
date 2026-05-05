# Finzo NexusChat

Enterprise RAG chatbot for Finzo Technologies.

## Features
- **Role-Based Access Control**: Different departments access different data.
- **HR CSV Analysis**: Pandas-powered querying of structured HR data.
- **PII Masking**: Automatically masks emails, phone numbers, and IDs.
- **Scope Guard**: Rejects queries unrelated to company knowledge.
- **Local Embeddings**: `sentence-transformers` running locally.

## Setup
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your API keys.
4. Place data files in their respective folders inside `data/` (e.g., `data/hr/hr_data.csv`).
5. Run ingestion: `python ingest.py`
6. Run the app: `streamlit run app.py`

## Users
Demo credentials are provided on the login page.
