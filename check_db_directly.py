"""
Direct PostgreSQL Database Check
Check what's actually stored in the database for image embeddings
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'research_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def check_database():
    """Check what's stored in the database"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )
        cursor = conn.cursor()
        
        print("=" * 70)
        print("DATABASE CONTENT CHECK - IMAGE EMBEDDINGS")
        print("=" * 70)
        print()
        
        # Check 1: What text was embedded for images
        print("📋 CHECK 1: Content Preview for Images")
        print("-" * 70)
        cursor.execute("""
            SELECT file_name, content_preview, embedding_dimension, file_type
            FROM document_embeddings 
            WHERE file_type = 'Image' 
            ORDER BY file_name 
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        for row in results:
            print(f"📄 {row['file_name']}")
            print(f"   Content: {row['content_preview']}")
            print(f"   Dimension: {row['embedding_dimension']}")
            print()
        
        print()
        print("=" * 70)
        print("📊 CHECK 2: Embedding Values Comparison")
        print("-" * 70)
        
        # Check 2: Compare actual embedding values
        cursor.execute("""
            SELECT file_name, embedding
            FROM document_embeddings 
            WHERE file_name IN ('puppy.jpg', 'german sheperd puppy.jpg', 'dollar.jpg')
            ORDER BY file_name
        """)
        
        results = cursor.fetchall()
        embeddings_data = {}
        
        for row in results:
            file_name = row['file_name']
            embedding = row['embedding']
            
            # Handle nested dict format
            if isinstance(embedding, dict) and 'embedding' in embedding:
                embedding = embedding['embedding']
            
            # Flatten if needed
            def flatten(lst):
                result = []
                for item in lst:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
            
            if isinstance(embedding, list):
                embedding = flatten(embedding)
            
            embeddings_data[file_name] = embedding
            
            print(f"\n📄 {file_name}")
            print(f"   Embedding length: {len(embedding)}")
            print(f"   First 10 values: {embedding[:10]}")
            print(f"   Last 10 values: {embedding[-10:]}")
            print(f"   Sum of values: {sum(embedding):.6f}")
            print(f"   Average value: {sum(embedding)/len(embedding):.6f}")
        
        # Check if embeddings are identical
        print()
        print("=" * 70)
        print("🔍 CHECK 3: Are Embeddings Different?")
        print("-" * 70)
        
        if len(embeddings_data) >= 2:
            files = list(embeddings_data.keys())
            emb1 = embeddings_data[files[0]]
            emb2 = embeddings_data[files[1]]
            
            # Calculate cosine similarity
            import numpy as np
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm1 * norm2)
            
            print(f"\n📊 Comparing: {files[0]} vs {files[1]}")
            print(f"   Cosine Similarity: {similarity:.6f}")
            
            if similarity > 0.95:
                print("   ⚠️  WARNING: Embeddings are TOO similar!")
                print("   This suggests they might be identical or nearly identical.")
            elif similarity > 0.80:
                print("   ⚠️  Embeddings are quite similar (might be an issue)")
            else:
                print("   ✅ Embeddings are sufficiently different")
        
        print()
        print("=" * 70)
        print("📝 CHECK 4: Sample Query Embedding")
        print("-" * 70)
        
        # Generate a query embedding for comparison
        print("\nGenerating embedding for query: 'dog'")
        
        import boto3
        session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', 'default'))
        bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {
                    "truncationMode": "END",
                    "value": "dog"
                }
            }
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=os.getenv('AWS_MODEL_ID', 'amazon.nova-2-multimodal-embeddings-v1:0'),
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        
        # Handle all possible response formats
        if 'embedding' in response_body:
            query_embedding = response_body['embedding']
            if isinstance(query_embedding, dict) and 'embedding' in query_embedding:
                query_embedding = query_embedding['embedding']
        elif 'embeddings' in response_body:
            query_embedding = response_body['embeddings'][0]
            if isinstance(query_embedding, dict) and 'embedding' in query_embedding:
                query_embedding = query_embedding['embedding']
        else:
            print(f"   ❌ Unexpected response format: {response_body.keys()}")
            print(f"   Response: {response_body}")
            return
        
        print(f"   Query embedding length: {len(query_embedding)}")
        print(f"   First 10 values: {query_embedding[:10]}")
        print(f"   Sum: {sum(query_embedding):.6f}")
        
        # Compare with puppy.jpg
        if 'puppy.jpg' in embeddings_data:
            puppy_emb = embeddings_data['puppy.jpg']
            vec_query = np.array(query_embedding)
            vec_puppy = np.array(puppy_emb)
            
            dot_product = np.dot(vec_query, vec_puppy)
            norm_query = np.linalg.norm(vec_query)
            norm_puppy = np.linalg.norm(vec_puppy)
            
            similarity = dot_product / (norm_query * norm_puppy)
            
            print(f"\n   Similarity between 'dog' query and 'puppy.jpg': {similarity:.6f}")
            print(f"   Expected: High similarity (>0.7)")
            print(f"   Actual: {'✅ Good' if similarity > 0.7 else '❌ Too low'}")
        
        cursor.close()
        conn.close()
        
        print()
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database()
