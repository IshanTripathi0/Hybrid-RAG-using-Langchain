# Hybrid-RAG-using-Langchain
Hybrid RAG refers to an advanced retrieval technique that combines vector similarity search with traditional search methods, such as full-text search or BM25. This approach enables more comprehensive and flexible information retrieval by leveraging the strengths of both methods, vector similarity for semantic understanding and traditional techniques for precise keyword or text-based matching.

# Usage
Clone the repository on your system
``` bash
git clone https://github.com/IshanTripathi0/Hybrid-RAG-using-Langchain.git
```
The repo has three files:
- **build_store**: Contains the code to build and store the vector database using the chosen embedding model
- **rag_pipeline**: Implements the core RAG pipeline for document retrieval and response generation
- **main.py**: Defines the query interface using gradio, processes input query and return the response using the rag pipeline

## Running the RAG app
- After cloning the repository, install the requirements:
  ```bash
  pip install -r requirements.txt
```
-
