import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """Establish and return a PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT", 5432),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD")
    )

def main():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        print("✅ Successfully connected to the database.")

        cursor.execute("SELECT file_name, content_preview, embedding FROM document_embeddings LIMIT 3;")
        rows = cursor.fetchall()
        if not rows:
            print("⚠️ No rows found in document_embeddings table.")
            return

        for row in rows:
            print("\nFile:", row[0])
            print("Text preview:", row[1][:200] if row[1] else "Empty")
            print("Embedding sample:", row[2][:5] if row[2] else "Empty")
            print("Embedding type:", type(row[2]))
            print("Embedding length:", len(row[2]) if row[2] else "Empty")

    except Exception as e:
        print("❌ Error:", e)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
            print("✅ Database connection closed.")

if __name__ == "__main__":
    main()
