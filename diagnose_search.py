"""
Diagnostic script to check database and search functionality
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'research_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def check_database():
    """Check database connection and data"""
    print("=" * 70)
    print("🔍 DATABASE DIAGNOSTICS")
    print("=" * 70)
    
    try:
        # Connect to database
        print(f"\n1️⃣ Connecting to database...")
        print(f"   Host: {DB_HOST}")
        print(f"   Port: {DB_PORT}")
        print(f"   Database: {DB_NAME}")
        print(f"   User: {DB_USER}")
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )
        
        print("   ✅ Connected successfully!")
        
        cursor = conn.cursor()
        
        # Check if table exists
        print(f"\n2️⃣ Checking if 'document_embeddings' table exists...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'document_embeddings'
            );
        """)
        table_exists = cursor.fetchone()['exists']
        
        if not table_exists:
            print("   ❌ Table 'document_embeddings' does not exist!")
            print("   💡 You need to run the data ingestion script first.")
            cursor.close()
            conn.close()
            return False
        
        print("   ✅ Table exists!")
        
        # Count total documents
        print(f"\n3️⃣ Counting documents in database...")
        cursor.execute("SELECT COUNT(*) as count FROM document_embeddings")
        total_count = cursor.fetchone()['count']
        print(f"   📊 Total documents: {total_count}")
        
        if total_count == 0:
            print("   ❌ No documents found in database!")
            print("   💡 You need to run the data ingestion script first.")
            cursor.close()
            conn.close()
            return False
        
        # Count documents with embeddings
        print(f"\n4️⃣ Checking documents with embeddings...")
        cursor.execute("SELECT COUNT(*) as count FROM document_embeddings WHERE embedding IS NOT NULL")
        embedded_count = cursor.fetchone()['count']
        print(f"   📊 Documents with embeddings: {embedded_count}")
        
        if embedded_count == 0:
            print("   ❌ No embeddings found!")
            print("   💡 Embeddings need to be generated.")
            cursor.close()
            conn.close()
            return False
        
        # Show file type distribution
        print(f"\n5️⃣ File type distribution...")
        cursor.execute("""
            SELECT file_type, COUNT(*) as count 
            FROM document_embeddings 
            GROUP BY file_type 
            ORDER BY count DESC
        """)
        file_types = cursor.fetchall()
        for ft in file_types:
            print(f"   📄 {ft['file_type']}: {ft['count']} files")
        
        # Show sample documents
        print(f"\n6️⃣ Sample documents (first 5)...")
        cursor.execute("""
            SELECT file_name, file_type, file_path, 
                   CASE WHEN embedding IS NOT NULL THEN 'Yes' ELSE 'No' END as has_embedding
            FROM document_embeddings 
            LIMIT 5
        """)
        samples = cursor.fetchall()
        for i, sample in enumerate(samples, 1):
            print(f"   {i}. {sample['file_name']}")
            print(f"      Type: {sample['file_type']}")
            print(f"      Embedding: {sample['has_embedding']}")
            print(f"      Path: {sample['file_path'][:60]}...")
        
        # Check embedding dimensions
        print(f"\n7️⃣ Checking embedding dimensions...")
        cursor.execute("""
            SELECT embedding_dimension, COUNT(*) as count
            FROM document_embeddings
            WHERE embedding IS NOT NULL
            GROUP BY embedding_dimension
        """)
        dimensions = cursor.fetchall()
        for dim in dimensions:
            print(f"   📐 Dimension {dim['embedding_dimension']}: {dim['count']} documents")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 70)
        print("✅ DATABASE CHECK COMPLETE - All systems operational!")
        print("=" * 70)
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Database connection failed!")
        print(f"   Error: {e}")
        print(f"\n💡 Possible solutions:")
        print(f"   1. Check if PostgreSQL is running")
        print(f"   2. Verify database credentials in .env file")
        print(f"   3. Ensure database '{DB_NAME}' exists")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search():
    """Test search functionality"""
    print("\n" + "=" * 70)
    print("🔍 SEARCH FUNCTIONALITY TEST")
    print("=" * 70)
    
    try:
        from query_service import keyword_search, semantic_search, hybrid_search
        
        # Test queries
        test_queries = [
            "drug",
            "insurance",
            "clinical trial",
            "medical research"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing query: '{query}'")
            print("-" * 70)
            
            # Test keyword search
            print(f"   🔤 Keyword Search...")
            keyword_results = keyword_search(query, top_k=3, min_score=0.0)
            print(f"      Found: {len(keyword_results)} results")
            if keyword_results:
                for i, r in enumerate(keyword_results[:2], 1):
                    print(f"      {i}. {r['file_name']} (score: {r['relevance_score']:.3f})")
            
            # Test semantic search
            print(f"   🧠 Semantic Search...")
            semantic_results = semantic_search(query, top_k=3, min_score=0.0)
            print(f"      Found: {len(semantic_results)} results")
            if semantic_results:
                for i, r in enumerate(semantic_results[:2], 1):
                    print(f"      {i}. {r['file_name']} (score: {r['relevance_score']:.3f})")
            
            # Test hybrid search
            print(f"   ⚡ Hybrid Search...")
            hybrid_results = hybrid_search(query, top_k=3, min_score=0.0)
            print(f"      Found: {len(hybrid_results)} results")
            if hybrid_results:
                for i, r in enumerate(hybrid_results[:2], 1):
                    print(f"      {i}. {r['file_name']} (score: {r['relevance_score']:.3f})")
        
        print("\n" + "=" * 70)
        print("✅ SEARCH TEST COMPLETE")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n❌ Cannot import query_service: {e}")
        print(f"   Make sure query_service.py is in the same directory")
    except Exception as e:
        print(f"\n❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all diagnostics"""
    print("\n🚀 Starting Research Intelligence Search Diagnostics...\n")
    
    # Check database
    db_ok = check_database()
    
    if db_ok:
        # Test search
        test_search()
    else:
        print("\n⚠️ Skipping search tests due to database issues")
    
    print("\n✅ Diagnostics complete!\n")

if __name__ == "__main__":
    main()
