import pickle
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv



def load_rag_pipeline(persist_dir="chroma_imdb_db", docs_path="split_docs.pkl"):
    load_dotenv()
    groq_api_key =os.getenv("GROQ_API_KEY")
    
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load persisted Chroma vectorstore
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # Dense retriever
    dense_retriever = vectorstore.as_retriever()

    # Load split documents from pickle
    try:
        with open(docs_path, "rb") as f:
            split_docs = pickle.load(f)
        print(f"Loaded {len(split_docs)} documents for BM25.")
    except FileNotFoundError:
        raise FileNotFoundError("split_docs.pkl not found. Please run build_store.py first.")

    # Keyword retriever using stored split docs
    keyword_retriever = BM25Retriever.from_documents(split_docs)
    keyword_retriever.k = 3

    # Hybrid retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the following context.
If you don't find the answer in the context, just say that you don't know.
Context: {context}

Question: {input}

Answer:
""")

    # Load LLM using Groq 
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

    #RAG pipeline
    rag_chain = (
        {"context": ensemble_retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

