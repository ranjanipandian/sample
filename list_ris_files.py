"""List all files in research intelligent search folder"""
import os

folder_path = r"D:\Users\2394071\Desktop\research intelligent search"

print("=" * 80)
print("FILES IN 'research intelligent search' FOLDER")
print("=" * 80)

all_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        full_path = os.path.join(root, file)
        all_files.append((file, full_path))

all_files.sort()

print(f"\nTotal files found: {len(all_files)}\n")

for i, (filename, filepath) in enumerate(all_files, 1):
    # Get relative path from base folder
    rel_path = filepath.replace(folder_path, "").lstrip("\\")
    print(f"{i}. {filename}")
    print(f"   Path: {rel_path}")

print("\n" + "=" * 80)
print(f"TOTAL: {len(all_files)} files")
print("=" * 80)
