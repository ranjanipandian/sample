"""Check actual paths in database"""
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
print("SAMPLE PATHS FROM DATABASE")
print("=" * 80)

cur.execute("""
    SELECT file_name, file_path 
    FROM document_embeddings 
    WHERE file_name IN ('puppy.jpg', 'insurance.csv', 'Ai to deep learning in drug discovery.pdf')
""")

results = cur.fetchall()

for file_name, file_path in results:
    print(f"\nFile: {file_name}")
    print(f"Path: {file_path}")
    
    pdf_path = '\\documents\\pdf\\'
    images_path = '\\visual_data\\images\\'
    csv_path = '\\embeddings\\structured_data\\csv\\'
    
    print(f"Contains '{pdf_path}': {pdf_path in file_path}")
    print(f"Contains '{images_path}': {images_path in file_path}")
    print(f"Contains '{csv_path}': {csv_path in file_path}")

cur.close()
conn.close()
