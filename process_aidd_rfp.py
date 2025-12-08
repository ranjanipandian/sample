"""
Process the 2021 AIDD RfP ARIBA.pdf file specifically
This will extract text, generate embeddings, and store in database
"""

import os
import json
import boto3
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2
import re

load_dotenv()

# Configuration
AWS_PROFILE = os.getenv('AWS_PROFILE', 'default')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_MODEL_ID = os.getenv('AWS_MODEL_ID', 'amazon.nova-2-multimodal-embeddings-v1:0')

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'research_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# File path
FILE_PATH = r"D:\Users\2394071\Desktop\research intelligent search\documents\pdf\2021 AIDD RfP ARIBA.pdf"

# Initialize AWS Bedrock client
session = boto3.Session(profile_name=AWS_PROFILE)
bedrock_runtime = session.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION
)

# Medical abbreviations and synonyms for normalization
MEDICAL_ABBREVIATIONS = {
    "BP": "blood pressure",
    "HR": "heart rate",
    "MI": "myocardial infarction",
    "DM": "diabetes mellitus",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
    "NSCLC": "non small cell lung cancer",
    "RFP": "request for proposal",
    "AIDD": "artificial intelligence drug discovery"
}

MEDICAL_SYNONYMS = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "low blood sugar": "hypoglycemia",
    "sugar": "glucose",
    "kidney failure": "renal failure",
    "blood sugar": "glucose level"
}

def normalize_text(text: str) -> str:
    """Normalize medical text for semantic embedding"""
    text = text.lower()
    # Replace abbreviations
    for abbr, full in MEDICAL_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr.lower()) + r'\b'
        text = re.sub(pattern, full.lower(), text)
    # Replace synonyms
    for syn, canonical in MEDICAL_SYNONYMS.items():
        pattern = r'\b' + re.escape(syn.lower()) + r'\b'
        text = re.sub(pattern, canonical.lower(), text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_path, max_chars=8000):
    """Extract text from PDF file"""
    try:
        print(f"📄 Extracting text from PDF...")
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            total_pages = len(pdf_reader.pages)
            print(f"   Total pages: {total_pages}")
            
            # Extract from first 10 pages or all pages if less
            pages_to_extract = min(10, total_pages)
            for page_num in range(pages_to_extract):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
                print(f"   Extracted page {page_num + 1}/{pages_to_extract}")
            
            print(f"   Total text extracted: {len(text)} characters")
            return text[:max_chars]
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return ""

def generate_embedding_text(text_content):
    """Generate embedding using Amazon Nova"""
    try:
        print(f"🧠 Generating embedding...")
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {
                    "truncationMode": "END",
                    "value": text_content[:8000]
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
        
        if embedding:
            print(f"✅ Embedding generated: {len(embedding)} dimensions")
        else:
            print(f"❌ No embedding returned from API")
            
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

def store_embedding(conn, file_data):
    """Store embedding in database"""
    try:
        print(f"💾 Storing in database...")
        cursor = conn.cursor()
        embedding_json = Json(file_data['embedding'])
        
        cursor.execute("""
        INSERT INTO document_embeddings
        (file_name, file_path, file_type, file_size, file_modified,
        embedding, embedding_dimension, content_preview, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (file_path)
        DO UPDATE SET
        file_name = EXCLUDED.file_name,
        file_type = EXCLUDED.file_type,
        file_size = EXCLUDED.file_size,
        file_modified = EXCLUDED.file_modified,
        embedding = EXCLUDED.embedding,
        embedding_dimension = EXCLUDED.embedding_dimension,
        content_preview = EXCLUDED.content_preview,
        updated_at = CURRENT_TIMESTAMP
        """, (
            file_data['file_name'],
            file_data['file_path'],
            file_data['file_type'],
            file_data['file_size'],
            file_data['file_modified'],
            embedding_json,
            file_data['embedding_dimension'],
            file_data['content_preview'],
            Json({'processed_date': datetime.now().isoformat()})
        ))
        
        conn.commit()
        cursor.close()
        print(f"✅ Successfully stored in database")
        return True
    except Exception as e:
        print(f"❌ Error storing embedding: {e}")
        conn.rollback()
        return False

def main():
    print("=" * 80)
    print("🔧 PROCESSING: 2021 AIDD RfP ARIBA.pdf")
    print("=" * 80)
    print()
    
    # Check if file exists
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return
    
    print(f"✅ File found: {FILE_PATH}")
    print()
    
    # Get file info
    file_name = os.path.basename(FILE_PATH)
    file_size = os.path.getsize(FILE_PATH)
    file_modified = datetime.fromtimestamp(os.path.getmtime(FILE_PATH))
    
    print(f"📊 File Information:")
    print(f"   Name: {file_name}")
    print(f"   Size: {file_size:,} bytes")
    print(f"   Modified: {file_modified}")
    print()
    
    # Extract text
    raw_text = extract_text_from_pdf(FILE_PATH)
    
    if not raw_text:
        print("❌ No text extracted from PDF")
        return
    
    print()
    print(f"📝 Text Preview (first 200 chars):")
    print(f"   {raw_text[:200]}...")
    print()
    
    # Normalize text
    print(f"🔄 Normalizing text...")
    normalized_text = normalize_text(raw_text)
    print(f"✅ Text normalized")
    print()
    
    # Generate embedding
    embedding = generate_embedding_text(normalized_text)
    
    if not embedding:
        print("❌ Failed to generate embedding")
        return
    
    print()
    
    # Connect to database
    try:
        print(f"🔌 Connecting to database...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"✅ Connected to database")
        print()
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return
    
    # Prepare file data
    file_data = {
        'file_name': file_name,
        'file_path': FILE_PATH,
        'file_type': 'pdf',
        'file_size': file_size,
        'file_modified': file_modified,
        'embedding': embedding,
        'content_preview': raw_text[:500],
        'embedding_dimension': len(embedding)
    }
    
    # Store in database
    success = store_embedding(conn, file_data)
    conn.close()
    
    print()
    print("=" * 80)
    if success:
        print("🎉 SUCCESS! File processed and indexed")
        print()
        print("✅ The file should now be searchable via:")
        print("   • Semantic search")
        print("   • Keyword search")
        print("   • Q&A system")
    else:
        print("❌ FAILED to process file")
    print("=" * 80)

if __name__ == "__main__":
    main()
