"""Check what files are in database and their actual paths"""
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
print("FILES IN DATABASE - Checking if they're in 'research intelligent search' folder")
print("=" * 80)

cur.execute("""
    SELECT file_name, file_path, file_type
    FROM document_embeddings 
    ORDER BY file_type, file_name
""")

results = cur.fetchall()

in_ris_folder = 0
not_in_ris_folder = 0

print("\nFILES IN 'research intelligent search' folder:")
print("-" * 80)
for row in results:
    file_name, file_path, file_type = row
    if 'research intelligent search' in file_path:
        print(f"✅ {file_type}: {file_name}")
        in_ris_folder += 1

print("\n" + "=" * 80)
print("\nFILES NOT IN 'research intelligent search' folder:")
print("-" * 80)
for row in results:
    file_name, file_path, file_type = row
    if 'research intelligent search' not in file_path:
        print(f"❌ {file_type}: {file_name}")
        print(f"   Path: {file_path}")
        not_in_ris_folder += 1

print("\n" + "=" * 80)
print(f"SUMMARY:")
print(f"  ✅ In 'research intelligent search': {in_ris_folder} files")
print(f"  ❌ NOT in 'research intelligent search': {not_in_ris_folder} files")
print(f"  📊 Total: {in_ris_folder + not_in_ris_folder} files")
print("=" * 80)

cur.close()
conn.close()
