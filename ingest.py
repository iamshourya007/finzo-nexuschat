"""Document and CSV loader for ChromaDB with metadata."""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def load_documents():
    """Load documents from data/ directory and attach department metadata."""
    documents = []
    
    base_dir = "data"
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # department tag is the folder name
            rel_path = os.path.relpath(root, base_dir)
            department = rel_path.split(os.sep)[0] if rel_path != "." else "general"
            
            if file.endswith(".md") or file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file.endswith(".csv"):
                loader = CSVLoader(file_path, encoding="utf-8")
            else:
                continue
                
            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["department"] = department
                    doc.metadata["source"] = file
                
                documents.extend(docs)
                print(f"Loaded {file} with department tag: {department}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
            
    return documents

def ingest_data():
    """Main ingestion pipeline."""
    docs = load_documents()
    if not docs:
        print("No documents found to ingest.")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Split into {len(splits)} chunks.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Saving to ChromaDB at {CHROMA_PERSIST_DIR}...")
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
