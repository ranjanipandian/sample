"""Check what content was embedded for images"""
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

print("=" * 80)
print("IMAGE FILES AND THEIR CONTENT")
print("=" * 80)

cur.execute("""
    SELECT file_name, content_preview 
    FROM document_embeddings 
    WHERE file_type='Image' 
    ORDER BY file_name
""")

results = cur.fetchall()

for file_name, content in results:
    print(f"\nFile: {file_name}")
    if content:
        print(f"Content: {content[:150]}...")
    else:
        print("Content: [No text content - image file]")

print("\n" + "=" * 80)
print(f"Total: {len(results)} image files")
print("=" * 80)

cur.close()
conn.close()
