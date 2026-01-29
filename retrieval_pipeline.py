import os
import sys
# Force UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def query_vector_store(query_text, persistent_directory="db/chroma_db", k=3):
    """
    Query the ChromaDB vector store for relevant documents using cosine similarity.
    """
    print(f"\n--- Querying Vector Store ---")
    print(f"Query: '{query_text}'")
    
    if not os.path.exists(persistent_directory):
        print(f"Error: Vector store not found at {persistent_directory}. Please run ingestion_pipeline.py first.")
        return

    # Initialize the same embedding model used in ingestion
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to the existing vector store
    # Note: We don't provide collection_metadata here as we are loading an existing one
    vectorstore = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model
    )
    
    print(f"Connected to vector store with {vectorstore._collection.count()} documents.")
    
    # Perform similarity search with score (cosine distance)
    # ChromaDB with "hnsw:space": "cosine" returns cosine distance.
    # Distance = 1 - Cosine Similarity. Lower distance = Higher similarity.
    print(f"Searching for top {k} relevant documents...")
    results = vectorstore.similarity_search_with_score(query_text, k=k)
    
    if not results:
        print("No relevant documents found.")
        return
        
    print(f"\nFound {len(results)} relevant chunks:")
    print("-" * 50)
    
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} (Cosine Distance: {score:.4f}):") 
        # Note: Score is distance, so lower is better. 
        # If you want Similarity, it would be roughly 1 - score (for normalized vectors)
        
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    # Define a new query as requested
    user_query = "What are the latest developments in artificial intelligence and machine learning?"
    
    query_vector_store(user_query)