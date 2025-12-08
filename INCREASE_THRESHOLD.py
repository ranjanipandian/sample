"""
Increase threshold for more precise semantic search results
This will reduce the number of results to only highly relevant documents
"""

import os

print("=" * 80)
print("🎯 ADJUSTING SEMANTIC SEARCH THRESHOLD")
print("=" * 80)
print()

print("📊 CURRENT SITUATION:")
print("-" * 80)
print("• Threshold: 0.45 (relatively low)")
print("• Result: 15-30 documents per query")
print("• Issue: Too many loosely related documents")
print()

print("💡 RECOMMENDED THRESHOLDS:")
print("-" * 80)
print("• 0.45 - Very broad search (current)")
print("• 0.55 - Balanced search (recommended)")
print("• 0.60 - Focused search (precise)")
print("• 0.70 - Very strict search (only exact matches)")
print()

# Read query_service.py
query_service_path = r"D:\Users\2394071\Desktop\multi-data retriever\query_service.py"

with open(query_service_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check current threshold
if 'min_score: float = 0.45' in content:
    print("✅ Found current threshold: 0.45")
    print()
    
    # Ask user for new threshold
    print("🎯 CHOOSE NEW THRESHOLD:")
    print("-" * 80)
    print("1. 0.55 - Balanced (10-15 results) - RECOMMENDED")
    print("2. 0.60 - Focused (5-10 results) - GOOD FOR DEMO")
    print("3. 0.65 - Strict (3-7 results)")
    print("4. 0.70 - Very strict (1-3 results)")
    print()
    
    choice = input("Enter choice (1-4) or press Enter for option 2 (0.60): ").strip()
    
    threshold_map = {
        '1': 0.55,
        '2': 0.60,
        '3': 0.65,
        '4': 0.70,
        '': 0.60  # default
    }
    
    new_threshold = threshold_map.get(choice, 0.60)
    
    print()
    print(f"📝 Setting new threshold to: {new_threshold}")
    print()
    
    # Update the file
    new_content = content.replace(
        'min_score: float = 0.45',
        f'min_score: float = {new_threshold}'
    )
    
    # Backup original
    backup_path = query_service_path + '.backup_threshold'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")
    
    # Write new version
    with open(query_service_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"✅ Updated query_service.py with threshold {new_threshold}")
    
    print()
    print("=" * 80)
    print("✅ THRESHOLD UPDATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("📋 NEXT STEPS:")
    print("-" * 80)
    print("1. Restart the Streamlit dashboard")
    print("2. Clear browser cache (Ctrl+Shift+Delete)")
    print("3. Test queries:")
    print("   • 'mushrooms' - should return 1-3 results")
    print("   • 'brain' - should return 1-5 results")
    print("   • 'clinical studies' - should return 3-7 results")
    print()
    print("💡 If you want to revert:")
    print(f"   Copy {backup_path} back to query_service.py")
    print()
    
else:
    print("⚠️ Could not find threshold setting in query_service.py")
    print("   Please check the file manually")
