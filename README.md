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
- You also need an API key to use a LLM. Here I used GroqAPI which is gives a lot of free usgae for a huge collection of their hosted LLMs, I recommend to use that if you don't already have it from
  [Groq](https://console.groq.com/keys). After creating you key save it in a .env file like this:
  ```bash
  GROQ_API_KEY = "your_api_key"
  ```
  and now you can load it using dotenv like this:
  ```bash
  pip install python-dotenv
  ```
  ```python
  import os
  from dotenv import load_dotenv
  load_dotenv()

  groq_api_key= os.getenv("GROQ_API_KEY")
  ```
- Open the build_store.py file to load your desired dataset from HuggingFace. Here I used the imdb movie review dataset from stanfordnlp but you can use some other, just make sure to load the correct path and content column.
  ```python
  loader = HuggingFaceDatasetLoader(
        path="imdb", #Replace the path with your own dataset (multiple datasets can also be used)
        page_content_column="text" #Choose the content column you want from the dataset
    )
  ```
  *Note: The Hybrid RAG like this one with BM25 retriever is best for datasets where the exact keyword is required (eg. Legal or Medical scenarios).*
  
- After selecting the dataset save and execute the file.
  ```bash
  python3.10 build_store.py
  ```
- Your vector store will be created at the specified storage path after a few minutes. Once that's done, you can run main.py to launch the Gradio app on localhost.
  ```bash
  python3.10 main.py
  ```
  
