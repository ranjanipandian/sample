"""
Check if the 2021 AIDD RfP ARIBA.pdf file is in the database
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🔍 CHECKING FOR 2021 AIDD RfP ARIBA.pdf")
print("=" * 80)
print()

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME'),
    cursor_factory=RealDictCursor
)

cursor = conn.cursor()

# Search for the specific file
print("1️⃣ Searching for exact file name...")
cursor.execute("""
    SELECT file_name, file_type, file_path, file_size, 
           CASE WHEN embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding,
           content_preview
    FROM document_embeddings
    WHERE file_name = '2021 AIDD RfP ARIBA.pdf'
""")

exact_match = cursor.fetchone()

if exact_match:
    print("✅ FOUND in database!")
    print(f"   File Type: {exact_match['file_type']}")
    print(f"   File Path: {exact_match['file_path']}")
    print(f"   File Size: {exact_match['file_size']} bytes")
    print(f"   Has Embedding: {exact_match['has_embedding']}")
    print(f"   Content Preview: {exact_match['content_preview'][:100] if exact_match['content_preview'] else 'None'}...")
else:
    print("❌ NOT FOUND in database!")
    print()
    print("2️⃣ Searching for similar file names...")
    cursor.execute("""
        SELECT file_name, file_type, file_path
        FROM document_embeddings
        WHERE LOWER(file_name) LIKE '%aidd%' OR LOWER(file_name) LIKE '%ariba%'
    """)
    
    similar = cursor.fetchall()
    if similar:
        print(f"   Found {len(similar)} similar files:")
        for s in similar:
            print(f"   • {s['file_name']}")
    else:
        print("   No similar files found")

print()
print("3️⃣ Checking all PDF files in database...")
cursor.execute("""
    SELECT file_name, 
           CASE WHEN embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding
    FROM document_embeddings
    WHERE file_type = 'pdf'
    ORDER BY file_name
""")

all_pdfs = cursor.fetchall()
print(f"   Total PDFs in database: {len(all_pdfs)}")
print()
print("   PDF files:")
for pdf in all_pdfs:
    status = "✅" if pdf['has_embedding'] == 'YES' else "❌"
    print(f"   {status} {pdf['file_name']}")

print()
print("=" * 80)
print("💡 DIAGNOSIS")
print("=" * 80)

if not exact_match:
    print()
    print("🔴 PROBLEM IDENTIFIED:")
    print("   The file '2021 AIDD RfP ARIBA.pdf' is NOT in the database!")
    print()
    print("🔧 SOLUTION:")
    print("   The file needs to be processed and added to the database.")
    print("   This can be done by running the embedding builder script.")
    print()
    print("📝 File location:")
    print("   D:\\Users\\2394071\\Desktop\\research intelligent search\\documents\\pdf\\2021 AIDD RfP ARIBA.pdf")
else:
    if exact_match['has_embedding'] == 'NO':
        print()
        print("🟡 PROBLEM IDENTIFIED:")
        print("   The file is in the database but has NO EMBEDDING!")
        print()
        print("🔧 SOLUTION:")
        print("   The embeddings need to be regenerated for this file.")
    elif not exact_match['content_preview']:
        print()
        print("🟡 PROBLEM IDENTIFIED:")
        print("   The file has an embedding but NO CONTENT PREVIEW!")
        print()
        print("🔧 SOLUTION:")
        print("   The text extraction needs to be re-run for this file.")
    else:
        print()
        print("✅ File is properly indexed with embedding and content!")
        print()
        print("🔍 If searches are still not working, the issue may be:")
        print("   1. Search query not matching the document content")
        print("   2. Similarity threshold too high")
        print("   3. Text extraction quality issues")

cursor.close()
conn.close()

print()
print("=" * 80)
