
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# -------------------- Load Environment --------------------
load_dotenv()

# Folder where the research documents are stored (with space in the name)
document_folder = r"D:\Users\2394071\Desktop\research intelligent search"  # Ensure this path is correct

# Load pre-trained sentence transformer model for semantic search
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# -------------------- Utility Functions --------------------
def load_documents_from_folder(folder_path):
    """Load text-based documents (CSV, TXT, etc.) from a folder and extract metadata (e.g., title, date)."""
    documents = []
    file_metadata = []  # To store metadata like title, date, etc.
    
    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Only process CSV and TXT files (exclude images, HTML, etc.)
        if filename.endswith(".csv") or filename.endswith(".txt"):  
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(file_path)  # Reading CSV into DataFrame
                elif filename.endswith(".txt"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    df = pd.DataFrame({"Content": [content]})  # Wrap text in a DataFrame for consistency

                # Assuming the document has columns like 'Title', 'Date', 'Content'
                for _, row in df.iterrows():
                    documents.append(row['Content'])  # 'Content' should be the column name of document text
                    file_metadata.append({
                        "file_name": filename,
                        "file_path": file_path,
                        "title": row.get("Title", ""),
                        "date": row.get("Date", ""),
                    })
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")

    print(f"Documents loaded: {len(documents)}")  # Debugging line to check the document count
    return documents, file_metadata

def semantic_search(query, documents, file_metadata, top_k=3, min_score=0.4):
    """Perform semantic search and return relevant documents with metadata."""
    if not documents:
        print("No documents available for search.")
        return []

    # Encode the query and documents into embeddings
    query_embedding = semantic_model.encode([query])  # Query embedding (2D array)
    document_embeddings = semantic_model.encode(documents)  # Document embeddings (2D array)

    # Check dimensions to ensure they are 2D arrays
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Document embeddings shape: {document_embeddings.shape}")

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, document_embeddings)
    
    # Get the top_k most similar documents
    most_similar_indices = np.argsort(similarities[0])[::-1][:top_k]
    
    # Return the top_k most relevant documents with metadata
    results = []
    for idx in most_similar_indices:
        if similarities[0][idx] >= min_score:
            results.append({
                "file_name": file_metadata[idx]["file_name"],
                "file_path": file_metadata[idx]["file_path"],
                "title": file_metadata[idx]["title"],
                "date": file_metadata[idx]["date"],
                "relevance_score": round(similarities[0][idx], 3),
            })
    
    return results

# -------------------- Main Query Service --------------------
if __name__ == "__main__":
    # Load documents and metadata from folder
    documents, file_metadata = load_documents_from_folder(document_folder)
    
    # Interactive Query Service
    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        search_type = input("Search type [semantic]: ").strip().lower() or "semantic"
        
        if search_type == "semantic":
            results = semantic_search(query, documents, file_metadata)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  Result {i}: {result['file_name']}")
                    print(f"   Path: {result['file_path']}")
                    print(f"   Title: {result['title']}")
                    print(f"   Date: {result['date']}")
                    print(f"   Relevance Score: {result['relevance_score']}")
            else:
                print("No relevant documents found.")
        else:
            print("Invalid search type.")
