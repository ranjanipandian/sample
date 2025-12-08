"""
Check what files are actually in the database
This will show why we're getting unexpected results
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 80)
print("🔍 CHECKING DATABASE CONTENTS")
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

# Get all files with embeddings
cursor.execute("""
    SELECT file_name, file_type, file_path, file_size
    FROM document_embeddings
    WHERE embedding IS NOT NULL
    ORDER BY file_type, file_name
""")

all_files = cursor.fetchall()

print(f"📊 Total files with embeddings: {len(all_files)}")
print()

# Group by file type
from collections import defaultdict
by_type = defaultdict(list)

for f in all_files:
    by_type[f['file_type']].append(f)

# Display by type
for file_type in sorted(by_type.keys()):
    files = by_type[file_type]
    print(f"\n{file_type}: {len(files)} files")
    print("-" * 80)
    for f in files:
        print(f"  • {f['file_name']}")

print()
print("=" * 80)
print("🔍 SPECIFIC FILE SEARCH")
print("=" * 80)

# Search for specific files mentioned by user
search_terms = ['mushroom', 'brain', 'clinical', 'trial']

for term in search_terms:
    print(f"\n📁 Files containing '{term}':")
    cursor.execute("""
        SELECT file_name, file_type, file_path
        FROM document_embeddings
        WHERE LOWER(file_name) LIKE %s
        AND embedding IS NOT NULL
        ORDER BY file_name
    """, (f'%{term}%',))
    
    results = cursor.fetchall()
    if results:
        for r in results:
            print(f"  • {r['file_name']} ({r['file_type']})")
    else:
        print(f"  (none found)")

cursor.close()
conn.close()

print()
print("=" * 80)
print("💡 ANALYSIS")
print("=" * 80)
print()
print("The semantic search is finding many documents because:")
print("1. It searches by MEANING, not just exact file names")
print("2. Documents with similar content get matched")
print("3. The threshold of 0.45 is relatively low")
print()
print("🎯 SOLUTIONS:")
print("1. Increase threshold to 0.60 for more precise results")
print("2. Use keyword search for exact matches")
print("3. Filter by file type in the query")
print()
