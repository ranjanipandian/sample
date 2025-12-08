"""
Diagnose database columns and data
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME', 'research_kb')

print("=" * 80)
print("🔍 DATABASE DIAGNOSIS")
print("=" * 80)
print()

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
    
    # Check table columns
    print("1️⃣ Checking table columns...")
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name='document_embeddings'
        ORDER BY ordinal_position
    """)
    
    columns = cursor.fetchall()
    print(f"   Found {len(columns)} columns:")
    for col in columns:
        print(f"   - {col['column_name']}: {col['data_type']}")
    
    print()
    print("2️⃣ Checking AIDD RfP entries...")
    cursor.execute("""
        SELECT file_name, file_path, file_type, 
               CASE WHEN file_modified IS NULL THEN 'NULL' ELSE 'HAS VALUE' END as file_modified_status,
               CASE WHEN content_preview IS NULL THEN 'NULL' ELSE 'HAS VALUE' END as content_status
        FROM document_embeddings
        WHERE file_name LIKE '%AIDD%'
    """)
    
    entries = cursor.fetchall()
    print(f"   Found {len(entries)} AIDD entries:")
    for entry in entries:
        print(f"   - {entry['file_name']}")
        print(f"     Type: {entry['file_type']}")
        print(f"     Path: {entry['file_path']}")
        print(f"     file_modified: {entry['file_modified_status']}")
        print(f"     content_preview: {entry['content_status']}")
        print()
    
    print("3️⃣ Testing safe query...")
    cursor.execute("""
        SELECT file_name, file_path, file_type, file_size, file_modified, content_preview
        FROM document_embeddings
        WHERE file_type NOT IN ('Image', 'Screenshot', 'HTML', 'Interactive Chart', 'HTML/Chart')
        ORDER BY COALESCE(file_modified, CURRENT_TIMESTAMP) DESC
        LIMIT 5
    """)
    
    rows = cursor.fetchall()
    print(f"   Query returned {len(rows)} rows")
    
    for i, row in enumerate(rows, 1):
        print(f"\n   Row {i}:")
        print(f"   - file_name: {row.get('file_name', 'MISSING')}")
        print(f"   - file_type: {row.get('file_type', 'MISSING')}")
        print(f"   - file_modified: {row.get('file_modified', 'MISSING')}")
        print(f"   - Keys in row: {list(row.keys())}")
    
    cursor.close()
    conn.close()
    
    print()
    print("=" * 80)
    print("✅ Diagnosis complete")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
