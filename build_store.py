# build_store.py

import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Config
CHROMA_DIR = "chroma_imdb_db"


def build_chroma_store():
    print("Loading IMDb dataset from Hugging Face...")
    loader = HuggingFaceDatasetLoader(
        path="imdb",
        page_content_column="text"
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} raw documents.")

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )
    split_documents = splitter.split_documents(documents)
    print(f"Created {len(split_documents)} chunks.")
    import pickle
    # Save the split documents for BM25
    with open("split_docs.pkl", "wb") as f:
    	pickle.dump(split_documents, f)
    print("Split documents saved to split_docs.pkl.")
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Creating Chroma vector store at: {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("Vector store built and saved successfully.")
    

if __name__ == "__main__":
    if os.path.exists(CHROMA_DIR):
        print(f"Vector store directory '{CHROMA_DIR}' already exists.")
        confirm = input("Overwrite it? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            exit(0)
        else:
            import shutil
            shutil.rmtree(CHROMA_DIR)
            print("Old vector store deleted.")

    build_chroma_store()

