"""
Database Setup Script for Research Intelligence Search Platform
This script creates the PostgreSQL database and schema for storing embeddings
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'research_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def create_database():
    """Create the database if it doesn't exist"""
    print("🔧 Step 1: Creating database...")
    
    try:
        # Connect to PostgreSQL server (default postgres database)
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"✅ Database '{DB_NAME}' created successfully!")
        else:
            print(f"ℹ️  Database '{DB_NAME}' already exists.")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def create_schema():
    """Create tables, indexes, and functions"""
    print("\n🔧 Step 2: Creating database schema...")
    
    try:
        # Connect to the research_kb database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        # Create extension
        print("  📦 Creating extensions...")
        cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        
        # Create document_embeddings table
        print("  📊 Creating document_embeddings table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                file_name VARCHAR(500) NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                file_type VARCHAR(50) NOT NULL,
                file_size BIGINT,
                file_modified TIMESTAMP,
                embedding JSONB NOT NULL,
                embedding_dimension INTEGER,
                metadata JSONB,
                content_preview TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        print("  🔍 Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON document_embeddings(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON document_embeddings(file_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_name ON document_embeddings(file_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON document_embeddings(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_gin ON document_embeddings USING GIN (embedding)")
        
        # Create search_history table
        print("  📝 Creating search_history table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_embedding JSONB,
                results_count INTEGER,
                search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_timestamp ON search_history(search_timestamp)")
        
        # Create system_metrics table
        print("  📈 Creating system_metrics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value NUMERIC,
                metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create cosine_similarity function
        print("  🧮 Creating cosine_similarity function...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION cosine_similarity(embedding1 JSONB, embedding2 JSONB)
            RETURNS FLOAT AS $$
            DECLARE
                dot_product FLOAT := 0;
                magnitude1 FLOAT := 0;
                magnitude2 FLOAT := 0;
                i INTEGER;
                arr1 FLOAT[];
                arr2 FLOAT[];
            BEGIN
                SELECT ARRAY(SELECT jsonb_array_elements_text(embedding1)::FLOAT) INTO arr1;
                SELECT ARRAY(SELECT jsonb_array_elements_text(embedding2)::FLOAT) INTO arr2;
                
                IF array_length(arr1, 1) != array_length(arr2, 1) THEN
                    RETURN 0;
                END IF;
                
                FOR i IN 1..array_length(arr1, 1) LOOP
                    dot_product := dot_product + (arr1[i] * arr2[i]);
                    magnitude1 := magnitude1 + (arr1[i] * arr1[i]);
                    magnitude2 := magnitude2 + (arr2[i] * arr2[i]);
                END LOOP;
                
                IF magnitude1 = 0 OR magnitude2 = 0 THEN
                    RETURN 0;
                END IF;
                
                RETURN dot_product / (sqrt(magnitude1) * sqrt(magnitude2));
            END;
            $$ LANGUAGE plpgsql IMMUTABLE
        """)
        
        # Create search_similar_documents function
        print("  🔍 Creating search_similar_documents function...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION search_similar_documents(
                query_embedding JSONB,
                similarity_threshold FLOAT DEFAULT 0.7,
                max_results INTEGER DEFAULT 10
            )
            RETURNS TABLE (
                id INTEGER,
                file_name VARCHAR(500),
                file_path TEXT,
                file_type VARCHAR(50),
                similarity_score FLOAT,
                content_preview TEXT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    de.id,
                    de.file_name,
                    de.file_path,
                    de.file_type,
                    cosine_similarity(de.embedding, query_embedding) as similarity_score,
                    de.content_preview
                FROM document_embeddings de
                WHERE cosine_similarity(de.embedding, query_embedding) >= similarity_threshold
                ORDER BY similarity_score DESC
                LIMIT max_results;
            END;
            $$ LANGUAGE plpgsql
        """)
        
        # Insert initial metrics
        print("  📊 Initializing system metrics...")
        cursor.execute("""
            INSERT INTO system_metrics (metric_name, metric_value) 
            VALUES 
                ('total_documents', 0),
                ('total_embeddings', 0),
                ('last_scan_timestamp', EXTRACT(EPOCH FROM NOW()))
            ON CONFLICT DO NOTHING
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database schema created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        return False

def verify_setup():
    """Verify the database setup"""
    print("\n🔧 Step 3: Verifying database setup...")
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print(f"  ✅ Found {len(tables)} tables:")
        for table in tables:
            print(f"     - {table[0]}")
        
        # Check functions
        cursor.execute("""
            SELECT routine_name 
            FROM information_schema.routines 
            WHERE routine_schema = 'public' AND routine_type = 'FUNCTION'
        """)
        functions = cursor.fetchall()
        print(f"  ✅ Found {len(functions)} functions:")
        for func in functions:
            print(f"     - {func[0]}")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Database setup verification complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying setup: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🧠 Research Intelligence Search Platform")
    print("   Database Setup Script")
    print("=" * 60)
    print()
    
    # Check if password is set
    if not DB_PASSWORD:
        print("❌ Error: DB_PASSWORD not found in .env file")
        return
    
    print(f"📋 Configuration:")
    print(f"   Host: {DB_HOST}")
    print(f"   Port: {DB_PORT}")
    print(f"   Database: {DB_NAME}")
    print(f"   User: {DB_USER}")
    print()
    
    # Step 1: Create database
    if not create_database():
        print("\n❌ Setup failed at database creation step")
        return
    
    # Step 2: Create schema
    if not create_schema():
        print("\n❌ Setup failed at schema creation step")
        return
    
    # Step 3: Verify setup
    if not verify_setup():
        print("\n❌ Setup verification failed")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Database setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run: python build_embeddings.py (to generate embeddings)")
    print("2. Run: streamlit run ris_app.py (to start dashboard)")
    print()

if __name__ == "__main__":
    main()
