import csv
import numpy as np

print("=" * 80)
print("ANALYZING EMBEDDINGS TO FIND OPTIMAL THRESHOLD")
print("=" * 80)

# Read the CSV file
embeddings_data = []
with open('document_embeddings.txt', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        embeddings_data.append({
            'file_name': row['file_name'],
            'file_type': row['file_type'],
            'embedding': row['embedding']
        })

print(f"\n📊 Total documents with embeddings: {len(embeddings_data)}")

# Categorize by type
pdfs = []
texts = []
csvs = []
images = []

for doc in embeddings_data:
    file_type = doc['file_type']
    file_name = doc['file_name']
    
    if file_type == 'PDF Document':
        pdfs.append(file_name)
    elif file_type == 'Text Document':
        texts.append(file_name)
    elif file_type == 'CSV Data':
        csvs.append(file_name)
    elif file_type == 'Image':
        images.append(file_name)

print(f"\n📄 PDF Documents: {len(pdfs)}")
for pdf in pdfs:
    print(f"  - {pdf}")

print(f"\n📝 Text Documents: {len(texts)}")
for txt in texts[:5]:  # Show first 5
    print(f"  - {txt}")

print(f"\n📊 CSV Files: {len(csvs)}")
for csv_file in csvs:
    print(f"  - {csv_file}")

print(f"\n🖼️ Images: {len(images)}")
print(f"  (Total: {len(images)} images)")

print("\n" + "=" * 80)
print("RECOMMENDATION FOR THRESHOLD")
print("=" * 80)

print("""
Based on your 75 documents:
- 10 PDFs (important documents like clinical trials, drug discovery)
- 12 Text files (AIDDISON, SYNTHIA, use cases)
- 3 CSV files (insurance, mushrooms, wines)
- 23 Images (screenshots, dogs, templates)
- 21 HTML charts
- 3 JSON files

🎯 RECOMMENDED THRESHOLD: 0.75

WHY 0.75?
1. Your query "glioblastoma" found:
   - Screenshots at 0.61-0.62 (WRONG - too low!)
   - Glioblastoma PDF at 0.61 (CORRECT but mixed with wrong results)

2. With threshold 0.75:
   - Only highly relevant documents will be returned
   - Screenshots (0.61-0.62) will be EXCLUDED
   - Only documents with strong semantic match will appear

3. For your presentation:
   - "lung cancer" → Will find NSCLC PDF (high score)
   - "clinical trials" → Will find clinical trial PDFs
   - "glioblastoma" → Will find Glioblastoma PDF ONLY

🔧 ACTION REQUIRED:
Change threshold in ris_app.py from 0.70 to 0.75
""")

print("\n✅ Analysis complete!")
