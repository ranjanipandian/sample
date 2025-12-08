"""
GPT-4o Semantic Search - Clean Implementation
Understands synonyms, abbreviations, context, and typos
Works for ALL files including RFPs
"""

import os
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

# Initialize GPT-4o client
try:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print(f"✅ GPT-4o Semantic Search initialized")
except Exception as e:
    print(f"❌ Failed to initialize GPT-4o: {e}")
    client = None

def get_all_documents():
    """Fetch all documents from database - SIMPLE AND SAFE"""
    documents = []
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )
        
        cursor = conn.cursor()
        
        # Simple query - just get all documents
        cursor.execute("""
            SELECT file_name, file_path, file_type, content_preview
            FROM document_embeddings
            WHERE content_preview IS NOT NULL
            ORDER BY file_name
        """)
        
        rows = cursor.fetchall()
        
        for row in rows:
            documents.append({
                'file_name': row['file_name'],
                'file_path': row['file_path'],
                'file_type': row['file_type'],
                'content_preview': row['content_preview'][:800]  # First 800 chars
            })
        
        cursor.close()
        conn.close()
        
        print(f"📊 Loaded {len(documents)} documents from database")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        import traceback
        traceback.print_exc()
        return []

def gpt4o_semantic_search(query: str, top_k: int = 10, min_score: float = 0.5) -> List[Dict[str, Any]]:
    """
    Use GPT-4o to perform intelligent semantic search
    Understands: synonyms, abbreviations, context, typos, medical terms
    """
    
    if not client:
        print("❌ GPT-4o client not available")
        return []
    
    print(f"\n🔍 GPT-4o Semantic Search: '{query}'")
    print(f"   Parameters: top_k={top_k}, min_score={min_score}")
    
    # Step 1: Get all documents
    documents = get_all_documents()
    
    if not documents:
        print("❌ No documents found in database")
        return []
    
    print(f"📚 Analyzing {len(documents)} documents...")
    
    # Step 2: Prepare document summaries for GPT-4o
    doc_summaries = []
    for idx, doc in enumerate(documents):
        doc_summaries.append({
            'id': idx,
            'file_name': doc['file_name'],
            'file_type': doc['file_type'],
            'content': doc['content_preview'][:500]  # First 500 chars for analysis
        })
    
    # Step 3: Use GPT-4o to rank documents with semantic understanding
    try:
        system_prompt = """You are an expert semantic search engine that understands:
- Medical terminology and synonyms (e.g., "brain tumor" = glioblastoma, brain cancer, neoplasm)
- Abbreviations (e.g., NGS = Next Generation Sequencing, RFP = Request for Proposal, AIDD = AI Drug Discovery)
- Context and related concepts (e.g., "fungus" relates to mushrooms, mycology)
- Typos and spelling variations
- Scientific and research terminology

Your task is to rank documents by semantic relevance, not just keyword matching.
Consider the MEANING and CONTEXT of the query, not just exact words.

Return JSON array of relevant documents (score >= 0.5):
[{"id": <doc_id>, "score": <0.0-1.0>, "reason": "<why relevant>"}]"""

        user_prompt = f"""Query: "{query}"

Documents to analyze:
{json.dumps(doc_summaries, indent=2)}

Rank these documents by semantic relevance to the query.

IMPORTANT GUIDELINES:
1. Understand synonyms: "brain tumor" matches "glioblastoma", "brain cancer"
2. Expand abbreviations: "NGS" matches "Next Generation Sequencing", "sequencing"
3. Consider context: "fungus" matches "mushroom", "mycology"
4. Handle typos: "diabetis" matches "diabetes"
5. Medical terms: "CHD" matches "coronary heart disease", "cardiac disease"
6. RFP terms: "AIDD" matches "AI drug discovery", "artificial intelligence"

Be INCLUSIVE - if a document is semantically related, include it even if exact keywords don't match.
Score based on semantic meaning, not keyword matching.

Return JSON array with relevant documents (score >= 0.5)."""

        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        llm_output = json.loads(response.choices[0].message.content)
        
        # Extract rankings flexibly
        rankings = None
        if isinstance(llm_output, list):
            rankings = llm_output
        elif isinstance(llm_output, dict):
            # Try common keys
            for key in ['rankings', 'documents', 'results', 'relevant_documents', 'ranked_documents']:
                if key in llm_output and isinstance(llm_output[key], list):
                    rankings = llm_output[key]
                    print(f"✅ Found rankings in key: '{key}'")
                    break
            
            # If still not found, take any list value
            if rankings is None:
                for key, value in llm_output.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and 'id' in value[0]:
                            rankings = value
                            print(f"✅ Found rankings in key: '{key}'")
                            break
        
        if not rankings:
            print(f"⚠️ Could not parse GPT-4o response")
            return []
        
        # Step 4: Build results
        results = []
        for ranking in rankings:
            doc_id = ranking.get('id')
            score = ranking.get('score', 0.0)
            reason = ranking.get('reason', '')
            
            if doc_id is not None and 0 <= doc_id < len(documents):
                doc = documents[doc_id]
                
                if score >= min_score:
                    results.append({
                        'file_name': doc['file_name'],
                        'file_path': doc['file_path'],
                        'file_type': doc['file_type'],
                        'content_preview': doc['content_preview'],
                        'relevance_score': round(float(score), 3),
                        'relevance_reason': reason,
                        'search_type': 'gpt4o_semantic'
                    })
        
        # Sort by score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = results[:top_k]
        
        print(f"✅ Found {len(results)} relevant documents")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in GPT-4o semantic search: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 GPT-4o Semantic Search Test")
    print("=" * 80)
    
    # Test queries with synonyms, abbreviations, context
    test_queries = [
        "brain tumor",
        "fungus",
        "NGS",
        "AIDD RfP",
        "coronary heart disease",
        "mushroom",
        "glioblastoma"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        results = gpt4o_semantic_search(query, top_k=5, min_score=0.5)
        
        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['file_name']}")
                print(f"   Type: {result['file_type']}")
                print(f"   Score: {result['relevance_score']:.3f}")
                print(f"   Reason: {result['relevance_reason']}")
        else:
            print("\n❌ No results found")
        
        print()
