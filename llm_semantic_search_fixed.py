"""
LLM-Based Semantic Search for Text Documents - FIXED VERSION
Uses GPT-4o to directly analyze and rank document relevance
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
from typing import List, Dict, Any

load_dotenv()

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'research_kb')

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
GPT4O_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print(f"✅ LLM Semantic Search (FIXED) initialized with GPT-4o")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Azure OpenAI client: {e}")
    client = None

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def fetch_documents_safe(limit=100):
    """Fetch text documents from database - SAFE VERSION"""
    documents = []
    
    try:
        conn = get_db_connection()
        if not conn:
            return documents
        
        cursor = conn.cursor()
        
        # Fetch documents with safe column access
        cursor.execute("""
            SELECT file_name, file_path, file_type, 
                   COALESCE(file_size, 0) as file_size,
                   file_modified,
                   COALESCE(content_preview, '') as content_preview
            FROM document_embeddings
            WHERE file_type NOT IN ('Image', 'Screenshot', 'HTML', 'Interactive Chart', 'HTML/Chart')
            AND LOWER(file_name) NOT LIKE '%.html'
            AND LOWER(file_name) NOT LIKE '%.htm'
            AND LOWER(file_name) NOT LIKE '%.png'
            AND LOWER(file_name) NOT LIKE '%.jpg'
            AND LOWER(file_name) NOT LIKE '%.jpeg'
            ORDER BY COALESCE(file_modified, CURRENT_TIMESTAMP) DESC
            LIMIT %s
        """, (limit,))
        
        rows = cursor.fetchall()
        
        # Safely build document list
        for row in rows:
            try:
                doc = {
                    'file_name': str(row['file_name']) if row['file_name'] else '',
                    'file_path': str(row['file_path']) if row['file_path'] else '',
                    'file_type': str(row['file_type']) if row['file_type'] else '',
                    'file_size': int(row['file_size']) if row['file_size'] else 0,
                    'file_modified': row['file_modified'],
                    'content_preview': str(row['content_preview']) if row['content_preview'] else ''
                }
                documents.append(doc)
            except Exception as e:
                print(f"⚠️ Skipping row due to error: {e}")
                continue
        
        cursor.close()
        conn.close()
        
        print(f"📊 Fetched {len(documents)} text documents")
        
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")
        import traceback
        traceback.print_exc()
    
    return documents

def rank_with_gpt4o(query: str, documents: List[Dict], top_k: int = 10):
    """Use GPT-4o to rank documents"""
    
    if not client:
        print("❌ GPT-4o client not available")
        return []
    
    if not documents:
        print("⚠️ No documents to rank")
        return []
    
    try:
        # Prepare summaries
        doc_summaries = []
        for idx, doc in enumerate(documents):
            doc_summaries.append({
                'id': idx,
                'file_name': doc['file_name'],
                'file_type': doc['file_type'],
                'content_preview': doc['content_preview'][:500]
            })
        
        # Call GPT-4o
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert at ranking documents by relevance. Return JSON array: [{\"id\": 0, \"score\": 0.95, \"reason\": \"...\"}]"},
                {"role": "user", "content": f"Query: \"{query}\"\n\nDocuments:\n{json.dumps(doc_summaries, indent=2)}\n\nRank by relevance (score 0.0-1.0). Include documents with score >= 0.5."}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        llm_output = json.loads(response.choices[0].message.content)
        
        # Extract rankings flexibly
        rankings = None
        if isinstance(llm_output, list):
            rankings = llm_output
        elif isinstance(llm_output, dict):
            for key in ['rankings', 'documents', 'results', 'relevant_documents']:
                if key in llm_output and isinstance(llm_output[key], list):
                    rankings = llm_output[key]
                    break
            if rankings is None:
                for value in llm_output.values():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and 'id' in value[0]:
                            rankings = value
                            break
        
        if not rankings:
            print("⚠️ Could not parse GPT-4o response")
            return []
        
        # Build results
        results = []
        for ranking in rankings:
            doc_id = ranking.get('id')
            score = ranking.get('score', 0.0)
            reason = ranking.get('reason', '')
            
            if doc_id is not None and 0 <= doc_id < len(documents):
                doc = documents[doc_id]
                results.append({
                    'file_name': doc['file_name'],
                    'file_path': doc['file_path'],
                    'file_type': doc['file_type'],
                    'file_size': doc['file_size'],
                    'file_modified': doc['file_modified'],
                    'content_preview': doc['content_preview'],
                    'relevance_score': round(float(score), 3),
                    'relevance_reason': reason,
                    'search_type': 'llm_semantic'
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        print(f"❌ Error in GPT-4o ranking: {e}")
        import traceback
        traceback.print_exc()
        return []

def semantic_search_fixed(query: str, top_k: int = 10, min_score: float = 0.6):
    """
    Perform LLM-based semantic search - FIXED VERSION
    """
    
    print(f"\n🔍 LLM Semantic Search (FIXED): '{query}'")
    print(f"   Parameters: top_k={top_k}, min_score={min_score}")
    
    # Fetch documents
    documents = fetch_documents_safe(limit=100)
    
    if not documents:
        print("⚠️ No documents found")
        return []
    
    # Rank with GPT-4o
    results = rank_with_gpt4o(query, documents, top_k=top_k)
    
    # Filter by score
    filtered = [r for r in results if r['relevance_score'] >= min_score]
    
    if not filtered and results:
        print(f"⚠️ No results above threshold {min_score}, returning top 3")
        filtered = results[:3]
    
    print(f"✅ Found {len(filtered)} results")
    
    return filtered

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Fixed Semantic Search")
    print("=" * 70)
    
    results = semantic_search_fixed("AIDD RfP", top_k=5, min_score=0.3)
    
    if results:
        print(f"\n✅ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['file_name']}")
            print(f"   Score: {result['relevance_score']:.3f}")
            print(f"   Reason: {result['relevance_reason']}")
    else:
        print("\n❌ No results found")
