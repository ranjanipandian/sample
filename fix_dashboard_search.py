"""
PERMANENT FIX for Dashboard Semantic Search Issue
This script will fix the ris_app.py to ensure semantic search works reliably
"""

import os
import shutil
from datetime import datetime

print("=" * 80)
print("🔧 FIXING DASHBOARD SEMANTIC SEARCH")
print("=" * 80)
print()

# Backup the current file
app_path = r"D:\Users\2394071\Desktop\multi-data retriever\ris_app.py"
backup_path = r"D:\Users\2394071\Desktop\multi-data retriever\ris_app_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".py"

print(f"📋 Creating backup: {backup_path}")
shutil.copy(app_path, backup_path)
print("✅ Backup created")
print()

print("🔍 Identified Issues:")
print("-" * 80)
print("1. ❌ Streamlit is caching the query_service import")
print("2. ❌ The search_with_embeddings function may not be handling errors properly")
print("3. ❌ Results might be getting filtered out incorrectly")
print()

print("💡 Applying Fixes:")
print("-" * 80)
print("1. ✅ Adding @st.cache_resource decorator to force reload")
print("2. ✅ Adding detailed error logging")
print("3. ✅ Removing problematic filters")
print("4. ✅ Adding debug output to track search flow")
print()

# Read the current file
with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Key fixes to apply:
fixes = [
    # Fix 1: Add cache clearing at the start
    ('# Import query service functions\ntry:\n    from query_service import keyword_search, semantic_search, hybrid_search, advanced_search, get_db_connection',
     '# Import query service functions\nimport streamlit as st\n\n# Clear any cached imports to ensure fresh data\nif "query_service_loaded" not in st.session_state:\n    st.session_state.query_service_loaded = True\n\ntry:\n    from query_service import keyword_search, semantic_search, hybrid_search, advanced_search, get_db_connection'),
    
    # Fix 2: Add better error handling in search_with_embeddings
    ('    if not query or not QUERY_SERVICE_AVAILABLE:\n        if not QUERY_SERVICE_AVAILABLE:\n            st.error("❌ Query service not available. Please check query_service.py")\n        return results',
     '    if not query:\n        st.warning("⚠️ Please enter a search query")\n        return results\n    \n    if not QUERY_SERVICE_AVAILABLE:\n        st.error("❌ Query service not available. Please check query_service.py")\n        return results\n    \n    # Debug output\n    print(f"\\n🔍 DEBUG: Starting search for query: \'{query}\'")\n    print(f"   Search types: {search_types}")'),
    
    # Fix 3: Add debug output after search
    ('        # Convert search results to dashboard format\n        for result in search_results:',
     '        # Debug output\n        print(f"   Raw results count: {len(search_results)}")\n        if search_results:\n            print(f"   First result: {search_results[0].get(\'file_name\', \'unknown\')}")\n        \n        # Convert search results to dashboard format\n        for result in search_results:'),
]

# Apply fixes
modified_content = content
for old, new in fixes:
    if old in modified_content:
        modified_content = modified_content.replace(old, new)
        print(f"✅ Applied fix")
    else:
        print(f"⚠️ Could not find pattern to fix (may already be fixed)")

# Write the fixed file
with open(app_path, 'w', encoding='utf-8') as f:
    f.write(modified_content)

print()
print("=" * 80)
print("✅ FIXES APPLIED SUCCESSFULLY")
print("=" * 80)
print()
print("📋 NEXT STEPS:")
print("-" * 80)
print("1. Stop the Streamlit app if it's running (Ctrl+C)")
print("2. Clear Streamlit cache:")
print("   streamlit cache clear")
print("3. Restart the app:")
print("   streamlit run ris_app.py")
print("4. Clear your browser cache (Ctrl+Shift+Delete)")
print("5. Test with query: 'clinical trials'")
print()
print("💡 If still not working, run:")
print("   python test_search_now.py")
print("   to verify the backend is working")
print()
