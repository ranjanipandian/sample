"""
LLM-Based Semantic Search for Text Documents
Uses GPT-4o to directly analyze and rank document relevance
For PDFs, Excel, CSV, Word documents, etc.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'research_kb')

# Azure OpenAI Configuration - USING GPT-4o (NOT GPT-4o Mini)
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
# Using GPT-4o deployment (full GPT-4o model, NOT mini)
GPT4O_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print(f"✅ LLM Text Semantic Search initialized with GPT-4o (deployment: {GPT4O_DEPLOYMENT})")
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

def fetch_text_documents(limit=100):
    """Fetch text documents from database"""
    documents = []
    
    try:
        conn = get_db_connection()
        if not conn:
            return documents
        
        cursor = conn.cursor()
        
        # Fetch documents (exclude images/HTML)
        cursor.execute("""
            SELECT file_name, file_path, file_type, file_size, file_modified, content_preview
            FROM document_embeddings
            WHERE file_type NOT IN ('Image', 'Screenshot', 'HTML', 'Interactive Chart', 'HTML/Chart')
            AND LOWER(file_name) NOT LIKE '%html'
            AND LOWER(file_name) NOT LIKE '%htm'
            AND LOWER(file_name) NOT LIKE '%png'
            AND LOWER(file_name) NOT LIKE '%jpg'
            AND LOWER(file_name) NOT LIKE '%jpeg'
            ORDER BY COALESCE(file_modified, CURRENT_TIMESTAMP) DESC
            LIMIT %s
        """, (limit,))
        
        rows = cursor.fetchall()
        
        for row in rows:
            documents.append({
                'file_name': row.get('file_name', ''),
                'file_path': row.get('file_path', ''),
                'file_type': row.get('file_type', ''),
                'file_size': row.get('file_size', 0),
                'file_modified': row.get('file_modified'),
                'content_preview': row.get('content_preview', '') or ''
            })
        
        cursor.close()
        conn.close()
        
        print(f"📊 Fetched {len(documents)} text documents from database")
        
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")
    
    return documents

def rank_documents_with_llm(query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Use GPT-4o to analyze and rank documents"""
    
    if not client:
        print("❌ Azure OpenAI client not available")
        return []
    
    if not documents:
        print("⚠️ No documents to rank")
        return []
    
    try:
        print(f"🤖 Using GPT-4o (deployment: {GPT4O_DEPLOYMENT}) to rank {len(documents)} documents for query: '{query}'")
        
        # Prepare document summaries
        doc_summaries = []
        for idx, doc in enumerate(documents):
            summary = {
                'id': idx,
                'file_name': doc['file_name'],
                'file_type': doc['file_type'],
                'content_preview': doc['content_preview'][:500] if doc['content_preview'] else ''
            }
            doc_summaries.append(summary)
        
        # System prompt
        system_prompt = """You are an expert at semantic document ranking for research and medical documents.
Your task is to analyze documents and determine their relevance to a user's query.

Consider:
1. Semantic meaning and intent of the query (understand synonyms, abbreviations, context)
2. Content relevance (not just keyword matching - understand medical concepts)
3. Document type and context
4. Medical/research terminology and related concepts
5. Spelling variations and common mistakes
6. Abbreviations and their expansions (e.g., CHD = coronary heart disease)

Be INCLUSIVE - if a document is even moderately relevant, include it.
Return a JSON array of document IDs with relevance scores (0.0 to 1.0).
Include documents with score >= 0.5 (moderately relevant or better).
Format: [{"id": 0, "score": 0.95, "reason": "brief explanation"}, ...]"""

        # User prompt
        user_prompt = f"""Query: "{query}"

Documents to rank:
{json.dumps(doc_summaries, indent=2)}

Analyze each document's relevance to the query. Consider:
- Semantic meaning (understand synonyms, abbreviations, context)
- Medical/research context and terminology
- Related concepts and terms (e.g., "heart disease" relates to "cardiac", "cardiovascular", "coronary")
- Document type appropriateness
- Content quality and specificity
- Spelling variations and common mistakes

IMPORTANT: Be INCLUSIVE - include documents that are moderately relevant or better.
Consider synonyms, related terms, and medical context.

Examples of semantic matching:
- "brain tumor" should match: glioblastoma, brain cancer, neoplasm, intracranial tumor
- "CHD" should match: coronary heart disease, cardiac disease, cardiovascular disease
- "diabetes" should match: diabetic, blood sugar, glucose, insulin

Return JSON array with relevant documents (score >= 0.5) sorted by relevance.
Format: [{{"id": <doc_id>, "score": <0.0-1.0>, "reason": "<brief explanation>"}}]

Be inclusive - return ALL moderately relevant documents."""

        # Call GPT-4o
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        llm_output = response.choices[0].message.content.strip()
        
        try:
            rankings = json.loads(llm_output)
            
            # Handle different JSON formats
            if isinstance(rankings, dict):
                if 'rankings' in rankings:
                    rankings = rankings['rankings']
                elif 'documents' in rankings:
                    rankings = rankings['documents']
                elif 'results' in rankings:
                    rankings = rankings['results']
                else:
                    for value in rankings.values():
                        if isinstance(value, list):
                            rankings = value
                            break
            
            if not isinstance(rankings, list):
                print(f"⚠️ Unexpected LLM response format: {type(rankings)}")
                return []
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response: {e}")
            return []
        
        # Build ranked results
        ranked_results = []
        for ranking in rankings:
            doc_id = ranking.get('id')
            score = ranking.get('score', 0.0)
            reason = ranking.get('reason', '')
            
            if doc_id is not None and 0 <= doc_id < len(documents):
                doc = documents[doc_id]
                ranked_results.append({
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
        
        # Sort by score and limit
        ranked_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        ranked_results = ranked_results[:top_k]
        
        print(f"✅ LLM ranked {len(ranked_results)} relevant documents")
        
        return ranked_results
        
    except Exception as e:
        print(f"❌ Error in LLM ranking: {e}")
        import traceback
        traceback.print_exc()
        return []

def llm_text_semantic_search(query: str, top_k: int = 10, min_score: float = 0.6) -> List[Dict[str, Any]]:
    """
    Perform LLM-based semantic search on text documents
    
    Args:
        query: User's search query
        top_k: Number of top results to return
        min_score: Minimum relevance score threshold
    
    Returns:
        List of ranked documents with relevance scores
    """
    
    print(f"\n🔍 LLM Text Semantic Search: '{query}'")
    print(f"   Parameters: top_k={top_k}, min_score={min_score}")
    
    # Fetch documents
    documents = fetch_text_documents(limit=100)
    
    if not documents:
        print("⚠️ No documents found in database")
        return []
    
    # Use LLM to rank documents
    ranked_results = rank_documents_with_llm(query, documents, top_k=top_k)
    
    # Filter by minimum score - be very lenient
    filtered_results = [r for r in ranked_results if r['relevance_score'] >= min_score]
    
    # If no results above threshold, return top results anyway (for presentation)
    if not filtered_results and ranked_results:
        print(f"⚠️ No results above threshold {min_score}, returning top {min(3, len(ranked_results))} results anyway")
        filtered_results = ranked_results[:min(3, len(ranked_results))]
    
    print(f"✅ Found {len(filtered_results)} results (threshold: {min_score})")
    
    return filtered_results

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 LLM-Based Text Semantic Search Test")
    print("=" * 70)
    
    # Test database connection
    conn = get_db_connection()
    if conn:
        print("✅ Database connection successful")
        conn.close()
    else:
        print("❌ Database connection failed")
        sys.exit(1)
    
    # Test queries
    test_queries = [
        "coronary heart disease treatment",
        "diabetes management",
        "cancer research"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)
        
        results = llm_text_semantic_search(query, top_k=5, min_score=0.6)
        
        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['file_name']}")
                print(f"   Type: {result['file_type']}")
                print(f"   Score: {result['relevance_score']:.3f}")
                print(f"   Reason: {result['relevance_reason']}")
        else:
            print("\n❌ No results found")
