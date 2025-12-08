"""
Embedding Generation Script for Research Intelligence Search Platform
Scans research intelligent search folder and generates embeddings using Amazon Nova-2
Includes text normalization for medical abbreviations, synonyms, and basic spelling corrections
"""

import os
import json
import boto3
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import base64
from tqdm import tqdm
import PyPDF2
from PIL import Image
import pandas as pd
from docx import Document
import re

# Load environment variables

load_dotenv()

# Configuration

AWS_PROFILE = os.getenv('AWS_PROFILE', 'default')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_MODEL_ID = os.getenv('AWS_MODEL_ID', 'amazon.nova-2-multimodal-embeddings-v1:0')
RESEARCH_DATA_PATH = os.getenv('RESEARCH_DATA_PATH', r'D:\Users\2394071\Desktop\research intelligent search')

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'research_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Initialize AWS Bedrock client

session = boto3.Session(profile_name=AWS_PROFILE)
bedrock_runtime = session.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION
)

# -------------------- Text Normalization --------------------

MEDICAL_ABBREVIATIONS = {
    "BP": "blood pressure",
    "HR": "heart rate",
    "MI": "myocardial infarction",
    "DM": "diabetes mellitus",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
    "NSCLC": "non small cell lung cancer"
}

MEDICAL_SYNONYMS = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "low blood sugar": "hypoglycemia",
    "sugar": "glucose",
    "kidney failure": "renal failure",
    "blood sugar": "glucose level"
}

def normalize_text(text: str) -> str:
    """Normalize medical text for semantic embedding"""
    text = text.lower()
    # Replace abbreviations
    for abbr, full in MEDICAL_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr.lower()) + r'\b'
        text = re.sub(pattern, full.lower(), text)
    # Replace synonyms
    for syn, canonical in MEDICAL_SYNONYMS.items():
        pattern = r'\b' + re.escape(syn.lower()) + r'\b'
        text = re.sub(pattern, canonical.lower(), text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------- File Extraction --------------------

def extract_text_from_pdf(file_path, max_chars=2000):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(min(3, len(pdf_reader.pages))):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text[:max_chars]
    except Exception as e:
        print(f"⚠️  Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_path, max_chars=2000):
    try:
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️  Error extracting DOCX text: {e}")
        return ""

def extract_text_from_txt(file_path, max_chars=2000):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()[:max_chars]
    except Exception as e:
        print(f"⚠️  Error extracting TXT text: {e}")
        return ""

def extract_data_from_excel(file_path, max_rows=10):
    try:
        df = pd.read_excel(file_path, nrows=max_rows)
        return df.to_string()[:2000]
    except Exception as e:
        print(f"⚠️  Error extracting Excel data: {e}")
        return ""

def extract_data_from_csv(file_path, max_rows=10):
    """Extract data from CSV file safely, skip bad rows"""
    try:
        df = pd.read_csv(file_path, nrows=max_rows, on_bad_lines='warn')  # skip bad lines
        return df.to_string()[:2000]
    except Exception as e:
        print(f"⚠️  Error extracting CSV data: {e}")
        return ""


def process_image(file_path):
    try:
        with Image.open(file_path) as img:
            max_size = (800, 800)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"⚠️  Error processing image: {e}")
        return None

# -------------------- Embedding Generation --------------------

def generate_embedding_text(text_content):
    try:
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {
                    "truncationMode": "END",
                    "value": text_content[:8000]
                }
            }
        }
        response = bedrock_runtime.invoke_model(
            modelId=AWS_MODEL_ID,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )
        body = json.loads(response['body'].read())
        embedding = None
        if 'embedding' in body:
            embedding = body['embedding']
            if isinstance(embedding, dict) and 'embedding' in embedding:
                embedding = embedding['embedding']
        elif 'embeddings' in body:
            embedding = body['embeddings'][0]
            if isinstance(embedding, dict) and 'embedding' in embedding:
                embedding = embedding['embedding']
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# -------------------- File Processing --------------------

def process_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

    content_preview = ""
    file_type = "unknown"
    embedding = None

    # Extract content
    if file_ext == '.pdf':
        file_type = "PDF Document"
        content_preview = extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        file_type = "Word Document"
        content_preview = extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        file_type = "Text Document"
        content_preview = extract_text_from_txt(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        file_type = "Excel Spreadsheet"
        content_preview = extract_data_from_excel(file_path)
    elif file_ext == '.csv':
        file_type = "CSV Data"
        content_preview = extract_data_from_csv(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        file_type = "Image"
        filename_text = os.path.splitext(file_name)[0].replace('_', ' ').replace('-', ' ')
        normalized_text = normalize_text(filename_text)
        embedding = generate_embedding_text(normalized_text)
        content_preview = f"Image file: {file_name}"
    elif file_ext in ['.html', '.htm', '.json']:
        file_type = file_ext.upper()
        content_preview = extract_text_from_txt(file_path)

    # Normalize and generate embedding for text files
    if content_preview and embedding is None:
        normalized_text = normalize_text(content_preview)
        embedding = generate_embedding_text(normalized_text)

    return {
        'file_name': file_name,
        'file_path': file_path,
        'file_type': file_type,
        'file_size': file_size,
        'file_modified': file_modified,
        'embedding': embedding,
        'content_preview': content_preview[:500],
        'embedding_dimension': len(embedding) if embedding else 0
    }

# -------------------- Database --------------------

def store_embedding(conn, file_data):
    try:
        cursor = conn.cursor()
        embedding_json = Json(file_data['embedding'])
        cursor.execute("""
        INSERT INTO document_embeddings
        (file_name, file_path, file_type, file_size, file_modified,
        embedding, embedding_dimension, content_preview, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (file_path)
        DO UPDATE SET
        file_name = EXCLUDED.file_name,
        file_type = EXCLUDED.file_type,
        file_size = EXCLUDED.file_size,
        file_modified = EXCLUDED.file_modified,
        embedding = EXCLUDED.embedding,
        embedding_dimension = EXCLUDED.embedding_dimension,
        content_preview = EXCLUDED.content_preview,
        updated_at = CURRENT_TIMESTAMP
        """, (
        file_data['file_name'],
        file_data['file_path'],
        file_data['file_type'],
        file_data['file_size'],
        file_data['file_modified'],
        embedding_json,
        file_data['embedding_dimension'],
        file_data['content_preview'],
        Json({})
        ))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"❌ Error storing embedding: {e}")
        conn.rollback()
        return False

# -------------------- Scan and Process --------------------

def scan_and_process_files():
    print("=" * 70)
    print("🧠 Research Intelligence Search Platform - Embedding Generation")
    print("=" * 70)

    if not os.path.exists(RESEARCH_DATA_PATH):
        print(f"❌ Research data path not found: {RESEARCH_DATA_PATH}")
        return

    INCLUDE_FOLDERS = ['documents', 'structured_data', 'visual_data', 'mixed_media']
    VALID_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.txt',
        '.xlsx', '.xls', '.csv', '.json',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
        '.html', '.htm'
    }

    all_files = []
    skipped_folders = []
    for root, dirs, files in os.walk(RESEARCH_DATA_PATH):
        should_include = any(f in root for f in INCLUDE_FOLDERS)
        if should_include:
            for file in files:
                if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:
                    all_files.append(os.path.join(root, file))
        else:
            if files and os.path.basename(root) not in skipped_folders:
                skipped_folders.append(os.path.basename(root))

    print(f"✅ Filtered folders: {', '.join(INCLUDE_FOLDERS)}")
    print(f"⚠️  Skipped folders: {', '.join(skipped_folders[:5])}{'...' if len(skipped_folders)>5 else ''}")
    print(f"📊 Found {len(all_files)} files to process")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME
        )
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return

    successful = 0
    failed = 0
    skipped = 0

    for file_path in tqdm(all_files, desc="Processing", unit="file"):
        try:
            file_data = process_file(file_path)
            if file_data['embedding']:
                if store_embedding(conn, file_data):
                    successful += 1
                else:
                    failed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            failed += 1

    conn.close()

    print("=" * 70)
    print("📊 Processing Summary")
    print("=" * 70)
    print(f"✅ Successfully processed: {successful}")
    print(f"⚠️  Skipped (no embedding): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total files: {len(all_files)}")
    print("🎉 Embedding generation complete!")

def main():
    scan_and_process_files()

if __name__ == "__main__":
    main()
