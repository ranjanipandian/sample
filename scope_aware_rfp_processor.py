"""
Scope-Aware RFP Document Processor
Processes RFP documents with metadata tagging for scope sections
Implements GPT-4o-based query classification and scope-prioritized search
"""

import os
import sys
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import AzureOpenAI
import boto3
from typing import List, Dict, Any, Tuple
import PyPDF2

# Load environment variables
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

# AWS Bedrock Configuration
AWS_PROFILE = os.getenv('AWS_PROFILE', 'default')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_MODEL_ID = os.getenv('AWS_MODEL_ID', 'amazon.nova-2-multimodal-embeddings-v1:0')

# Initialize clients
try:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print(f"✅ GPT-4o client initialized (deployment: {GPT4O_DEPLOYMENT})")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize GPT-4o client: {e}")
    openai_client = None

try:
    session = boto3.Session(profile_name=AWS_PROFILE)
    bedrock_runtime = session.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION
    )
    print("✅ AWS Bedrock client initialized")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize AWS Bedrock client: {e}")
    bedrock_runtime = None

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

def extract_pdf_text_by_pages(pdf_path: str) -> Dict[int, str]:
    """Extract text from PDF page by page"""
    page_texts = {}
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            print(f"📄 Extracting text from {total_pages} pages...")
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                page_texts[page_num + 1] = text  # 1-indexed pages
                
        print(f"✅ Extracted text from {len(page_texts)} pages")
        return page_texts
        
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return {}

def identify_scope_sections(page_texts: Dict[int, str]) -> Dict[int, bool]:
    """
    Use GPT-4o to identify which pages contain scope-related content
    For 2021 AIDD RFP: Scope is on pages 4-7
    """
    
    if not openai_client:
        # Fallback: Manual identification for known RFP
        print("⚠️ GPT-4o not available, using manual scope identification")
        scope_pages = {
            1: False, 2: False, 3: False,
            4: True, 5: True, 6: True, 7: True,  # Scope pages
            8: False, 9: False, 10: False
        }
        return scope_pages
    
    scope_pages = {}
    
    try:
        print("🤖 Using GPT-4o to identify scope sections...")
        
        for page_num, text in page_texts.items():
            # Use GPT-4o to classify if page contains scope content
            prompt = f"""Analyze this page from an RFP document and determine if it contains SCOPE-related content.

Scope content includes:
- Project scope and objectives
- Core capabilities and functional requirements
- Technical requirements and architecture
- Product features and modules
- Required services and deliverables
- Implementation details

Page {page_num} content:
{text[:1000]}...

Is this page part of the SCOPE section? Answer with just "YES" or "NO"."""

            response = openai_client.chat.completions.create(
                model=GPT4O_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing RFP documents and identifying scope sections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            scope_pages[page_num] = (answer == "YES")
            
        print(f"✅ Identified {sum(scope_pages.values())} scope pages out of {len(scope_pages)} total pages")
        return scope_pages
        
    except Exception as e:
        print(f"❌ Error identifying scope sections: {e}")
        # Fallback for 2021 AIDD RFP
        return {i: (4 <= i <= 7) for i in range(1, 11)}

def generate_embedding_with_nova(text: str) -> List[float]:
    """Generate embedding using Amazon Nova-2"""
    
    if not bedrock_runtime:
        print("❌ AWS Bedrock client not available")
        return None
    
    try:
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {
                    "truncationMode": "END",
                    "value": text[:8000]
                }
            }
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=AWS_MODEL_ID,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )
        
        body = json.loads(response['body'].read())
        embedding = None
        
        if 'embedding' in body:
            embedding = body['embedding']
            if isinstance(embedding, dict) and 'embedding' in embedding:
                embedding = embedding['embedding']
        elif 'embeddings' in body:
            embedding = body['embeddings'][0]
            if isinstance(embedding, dict) and 'embedding' in embedding:
                embedding = embedding['embedding']
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

def store_rfp_with_metadata(pdf_path: str, file_name: str):
    """
    Process RFP document and store with scope metadata
    Strategy: Store full document with metadata tags for scope sections
    """
    
    print(f"\n{'='*70}")
    print(f"📋 Processing RFP: {file_name}")
    print(f"{'='*70}")
    
    # Step 1: Extract text by pages
    page_texts = extract_pdf_text_by_pages(pdf_path)
    if not page_texts:
        print("❌ Failed to extract PDF text")
        return False
    
    # Step 2: Identify scope sections
    scope_pages = identify_scope_sections(page_texts)
    
    # Step 3: Create full document text
    full_text = "\n\n".join([f"[Page {num}]\n{text}" for num, text in page_texts.items()])
    
    # Step 4: Create scope-only text
    scope_text = "\n\n".join([f"[Page {num}]\n{text}" for num, text in page_texts.items() if scope_pages.get(num, False)])
    
    # Step 5: Generate embeddings
    print("\n🔄 Generating embeddings...")
    full_embedding = generate_embedding_with_nova(full_text)
    scope_embedding = generate_embedding_with_nova(scope_text)
    
    if not full_embedding or not scope_embedding:
        print("❌ Failed to generate embeddings")
        return False
    
    # Step 6: Store in database with metadata
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Check if table has metadata columns, if not add them
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='document_embeddings' AND column_name='is_scope_section'
        """)
        
        if not cursor.fetchone():
            print("📊 Adding metadata columns to database...")
            cursor.execute("""
                ALTER TABLE document_embeddings 
                ADD COLUMN IF NOT EXISTS is_scope_section BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS scope_pages TEXT,
                ADD COLUMN IF NOT EXISTS document_type VARCHAR(50)
            """)
            conn.commit()
        
        # Store full document with metadata
        scope_pages_json = json.dumps(scope_pages)
        
        cursor.execute("""
            INSERT INTO document_embeddings 
            (file_name, file_path, file_type, content_preview, embedding, embedding_dimension, 
             is_scope_section, scope_pages, document_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) 
            DO UPDATE SET 
                content_preview = EXCLUDED.content_preview,
                embedding = EXCLUDED.embedding,
                is_scope_section = EXCLUDED.is_scope_section,
                scope_pages = EXCLUDED.scope_pages,
                document_type = EXCLUDED.document_type
        """, (
            file_name,
            pdf_path,
            'RFP Document',
            full_text[:500],
            json.dumps(full_embedding),
            1024,
            False,  # This is the full document
            scope_pages_json,
            'RFP'
        ))
        
        # Store scope-only version with metadata
        cursor.execute("""
            INSERT INTO document_embeddings 
            (file_name, file_path, file_type, content_preview, embedding, embedding_dimension, 
             is_scope_section, scope_pages, document_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) 
            DO UPDATE SET 
                content_preview = EXCLUDED.content_preview,
                embedding = EXCLUDED.embedding,
                is_scope_section = EXCLUDED.is_scope_section,
                scope_pages = EXCLUDED.scope_pages,
                document_type = EXCLUDED.document_type
        """, (
            f"{file_name} (SCOPE ONLY)",
            f"{pdf_path}_scope",
            'RFP Document - Scope',
            scope_text[:500],
            json.dumps(scope_embedding),
            1024,
            True,  # This is scope-only
            scope_pages_json,
            'RFP_SCOPE'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n✅ Successfully stored RFP with scope metadata")
        print(f"   - Full document: {len(full_text)} characters")
        print(f"   - Scope section: {len(scope_text)} characters")
        print(f"   - Scope pages: {[p for p, is_scope in scope_pages.items() if is_scope]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing RFP: {e}")
        import traceback
        traceback.print_exc()
        return False

def classify_query_intent(query: str) -> Tuple[bool, str]:
    """
    Use GPT-4o to classify if query is scope-related
    Returns: (is_scope_query, explanation)
    """
    
    if not openai_client:
        print("⚠️ GPT-4o not available, assuming scope query")
        return True, "GPT-4o not available"
    
    try:
        prompt = f"""Analyze this query and determine if it's asking about the SCOPE of an RFP project.

Scope-related queries ask about:
- Project objectives and goals
- Core capabilities and features
- Technical requirements
- Functional modules
- Required services
- Implementation details
- Architecture and design
- Product roadmap
- Deliverables

Non-scope queries ask about:
- Company information
- Legal terms
- Privacy policies
- Response instructions
- Timelines (unless asking about scope timeline)
- General RFP process

Query: "{query}"

Is this a scope-related query? Respond in JSON format:
{{"is_scope_query": true/false, "explanation": "brief reason"}}"""

        response = openai_client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing RFP queries and classifying their intent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        is_scope = result.get('is_scope_query', True)
        explanation = result.get('explanation', '')
        
        print(f"🎯 Query classification: {'SCOPE' if is_scope else 'NON-SCOPE'} - {explanation}")
        
        return is_scope, explanation
        
    except Exception as e:
        print(f"❌ Error classifying query: {e}")
        return True, "Error in classification"

def scope_aware_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform scope-aware search:
    1. Classify query intent
    2. Prioritize scope sections if scope-related
    3. Use GPT-4o for ranking
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 Scope-Aware Search: '{query}'")
    print(f"{'='*70}")
    
    # Step 1: Classify query
    is_scope_query, explanation = classify_query_intent(query)
    
    # Step 2: Fetch documents
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        if is_scope_query:
            # Prioritize scope documents
            print("📋 Prioritizing SCOPE sections...")
            cursor.execute("""
                SELECT file_name, file_path, file_type, content_preview, 
                       is_scope_section, scope_pages, document_type
                FROM document_embeddings
                WHERE document_type IN ('RFP', 'RFP_SCOPE')
                ORDER BY is_scope_section DESC, file_modified DESC
            """)
        else:
            # Search all RFP content
            print("📄 Searching ALL RFP content...")
            cursor.execute("""
                SELECT file_name, file_path, file_type, content_preview, 
                       is_scope_section, scope_pages, document_type
                FROM document_embeddings
                WHERE document_type IN ('RFP', 'RFP_SCOPE')
                ORDER BY file_modified DESC
            """)
        
        documents = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not documents:
            print("⚠️ No RFP documents found in database")
            return []
        
        print(f"📊 Found {len(documents)} RFP documents")
        
        # Step 3: Use GPT-4o to rank documents
        results = rank_documents_with_gpt4o(query, documents, is_scope_query, top_k)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in scope-aware search: {e}")
        import traceback
        traceback.print_exc()
        return []

def rank_documents_with_gpt4o(query: str, documents: List[Dict], is_scope_query: bool, top_k: int) -> List[Dict[str, Any]]:
    """Use GPT-4o to rank documents with scope awareness"""
    
    if not openai_client:
        print("❌ GPT-4o not available")
        return []
    
    try:
        # Prepare document summaries
        doc_summaries = []
        for idx, doc in enumerate(documents):
            summary = {
                'id': idx,
                'file_name': doc['file_name'],
                'file_type': doc['file_type'],
                'is_scope': doc.get('is_scope_section', False),
                'content_preview': doc['content_preview'][:500] if doc['content_preview'] else ''
            }
            doc_summaries.append(summary)
        
        scope_instruction = ""
        if is_scope_query:
            scope_instruction = """
IMPORTANT: This is a SCOPE-related query. Prioritize documents marked as 'is_scope: true'.
Give higher relevance scores to scope sections (typically 0.8-1.0 for highly relevant scope content).
"""
        
        prompt = f"""Analyze these RFP documents and rank them by relevance to the query.

Query: "{query}"

{scope_instruction}

Documents:
{json.dumps(doc_summaries, indent=2)}

Rank documents by relevance. Consider:
- Semantic meaning and context
- Scope vs non-scope content (if scope query)
- Content quality and specificity

Return JSON array with relevant documents (score >= 0.5):
[{{"id": <doc_id>, "score": <0.0-1.0>, "reason": "<brief explanation>"}}]"""

        response = openai_client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and ranking RFP documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        llm_output = json.loads(response.choices[0].message.content)
        
        # Debug: Print the actual response structure
        print(f"🔍 GPT-4o Response Keys: {list(llm_output.keys()) if isinstance(llm_output, dict) else 'Not a dict'}")
        
        # Handle different JSON formats - be very flexible
        rankings = None
        
        if isinstance(llm_output, list):
            # Direct list format
            rankings = llm_output
        elif isinstance(llm_output, dict):
            # Try common keys first
            for key in ['rankings', 'documents', 'results', 'relevant_documents', 'ranked_documents', 'document_rankings']:
                if key in llm_output and isinstance(llm_output[key], list):
                    rankings = llm_output[key]
                    print(f"✅ Found rankings in key: '{key}'")
                    break
            
            # If still not found, look for ANY list value
            if rankings is None:
                for key, value in llm_output.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if it looks like a ranking (has 'id' and 'score')
                        if isinstance(value[0], dict) and 'id' in value[0]:
                            rankings = value
                            print(f"✅ Found rankings in key: '{key}'")
                            break
        
        if rankings is None or not isinstance(rankings, list):
            print(f"⚠️ Could not find rankings list in response")
            print(f"Response structure: {json.dumps(llm_output, indent=2)[:800]}")
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
                    'content_preview': doc['content_preview'],
                    'is_scope_section': doc.get('is_scope_section', False),
                    'relevance_score': round(float(score), 3),
                    'relevance_reason': reason,
                    'search_type': 'scope_aware_gpt4o'
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = results[:top_k]
        
        print(f"✅ Ranked {len(results)} relevant documents")
        
        return results
        
    except Exception as e:
        print(f"❌ Error ranking documents: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("="*70)
    print("📋 Scope-Aware RFP Processor")
    print("="*70)
    
    # Process the 2021 AIDD RFP
    rfp_path = r"D:\Users\2394071\Desktop\research intelligent search\documents\pdf\2021 AIDD RfP ARIBA.pdf"
    
    if os.path.exists(rfp_path):
        print(f"\n📄 Processing RFP: {rfp_path}")
        success = store_rfp_with_metadata(rfp_path, "2021 AIDD RfP ARIBA.pdf")
        
        if success:
            print("\n" + "="*70)
            print("✅ RFP processed successfully!")
            print("="*70)
            
            # Test with sample queries
            test_queries = [
                "What are the core capabilities?",
                "What is the project scope?",
                "What are the technical requirements?",
                "What is Merck's privacy policy?",
                "What is the timeline for go-live?"
            ]
            
            print("\n" + "="*70)
            print("🧪 Testing Scope-Aware Search")
            print("="*70)
            
            for query in test_queries:
                results = scope_aware_search(query, top_k=3)
                
                if results:
                    print(f"\n✅ Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['file_name']}")
                        print(f"   Score: {result['relevance_score']:.3f}")
                        print(f"   Scope: {'YES' if result['is_scope_section'] else 'NO'}")
                        print(f"   Reason: {result['relevance_reason']}")
                else:
                    print("\n❌ No results found")
    else:
        print(f"❌ RFP file not found: {rfp_path}")
