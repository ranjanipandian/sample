"""
Integrated Text Extraction + Embedding Generation Pipeline
Extracts text from all files, then generates embeddings using Amazon Nova-2
"""

import os
import json
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from datetime import datetime
from text_extractor import extract_text, process_all_files

# Load environment variables
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

# Initialize AWS Bedrock client
session = boto3.Session(profile_name=AWS_PROFILE)
bedrock_runtime = session.client("bedrock-runtime", region_name=AWS_REGION)

# Database connection
def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursor_factory=RealDictCursor
    )

def generate_embedding(text_content):
    """Generate embedding using Amazon Nova-2"""
    try:
        # Truncate text to 8000 characters for Nova-2
        text_content = text_content[:8000]
        
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {
                    "truncationMode": "END",
                    "value": text_content
                }
            }
        }

        response = bedrock_runtime.invoke_model(
            modelId=AWS_MODEL_ID,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )

        body = json.loads(response["body"].read())

        # Handle response format
        if "embedding" in body:
            emb = body["embedding"]
            if isinstance(emb, dict) and "embedding" in emb:
                return emb["embedding"]
            return emb
        elif "embeddings" in body:
            return body["embeddings"][0].get("embedding")
        
        return None

    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

def update_database_with_text_and_embeddings(folder_path):
    """
    Main pipeline:
    1. Extract text from all files
    2. Generate embeddings using Amazon Nova-2
    3. Update database with text and embeddings
    """
    print("=" * 80)
    print("🚀 INTEGRATED TEXT EXTRACTION + EMBEDDING PIPELINE")
    print("=" * 80)
    
    # Step 1: Extract text from all files
    print("\n📄 STEP 1: Extracting text from all files...")
    extracted_docs = process_all_files(folder_path)
    
    if not extracted_docs:
        print("❌ No text extracted from any files!")
        return
    
    print(f"\n✅ Extracted text from {len(extracted_docs)} files")
    
    # Step 2: Connect to database
    print("\n🔌 STEP 2: Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Step 3: Update each document with extracted text and new embeddings
    print("\n🧠 STEP 3: Generating embeddings and updating database...")
    
    updated_count = 0
    failed_count = 0
    
    for file_path, extracted_text in extracted_docs.items():
        try:
            file_name = os.path.basename(file_path)
            
            # Generate embedding from extracted text
            print(f"\n📝 Processing: {file_name}")
            print(f"   Text length: {len(extracted_text)} characters")
            
            embedding = generate_embedding(extracted_text)
            
            if embedding:
                # Update database with extracted text and new embedding
                cursor.execute("""
                    UPDATE document_embeddings
                    SET content_preview = %s,
                        embedding = %s,
                        updated_at = %s
                    WHERE file_path = %s
                """, (
                    extracted_text[:500],  # Store first 500 chars as preview
                    json.dumps(embedding),  # Store as JSONB
                    datetime.now(),
                    file_path
                ))
                
                updated_count += 1
                print(f"   ✅ Updated with new embedding")
            else:
                failed_count += 1
                print(f"   ❌ Failed to generate embedding")
                
        except Exception as e:
            failed_count += 1
            print(f"   ❌ Error: {e}")
    
    # Commit changes
    conn.commit()
    cursor.close()
    conn.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PIPELINE COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully updated: {updated_count} documents")
    print(f"❌ Failed: {failed_count} documents")
    print(f"📁 Total processed: {len(extracted_docs)} documents")
    print("\n🎉 Database updated with extracted text and new embeddings!")
    print("🔍 Semantic search will now work based on actual document content!")

if __name__ == "__main__":
    # Main folder to process
    folder_path = r"D:\Users\2394071\Desktop\research intelligent search"
    
    print("🚀 Starting integrated text extraction + embedding pipeline...")
    print(f"📁 Target folder: {folder_path}")
    
    confirm = input("\n⚠️ This will update all embeddings in the database. Continue? (yes/no): ")
    
    if confirm.lower() == 'yes':
        update_database_with_text_and_embeddings(folder_path)
    else:
        print("❌ Operation cancelled")
