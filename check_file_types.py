"""Check what file types are in the database"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
)

cur = conn.cursor()

print("=" * 70)
print("FILE TYPES IN DATABASE")
print("=" * 70)

cur.execute("""
    SELECT file_type, COUNT(*) as count 
    FROM document_embeddings 
    GROUP BY file_type 
    ORDER BY count DESC
""")

results = cur.fetchall()
total = 0

for row in results:
    print(f"{row[0]}: {row[1]} files")
    total += row[1]

print("=" * 70)
print(f"TOTAL: {total} files")
print("=" * 70)

# Show sample files from each type
print("\nSAMPLE FILES BY TYPE:")
print("=" * 70)

for file_type, count in results:
    print(f"\n{file_type} ({count} files):")
    cur.execute("""
        SELECT file_name 
        FROM document_embeddings 
        WHERE file_type = %s 
        LIMIT 3
    """, (file_type,))
    
    samples = cur.fetchall()
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample[0]}")

cur.close()
conn.close()
