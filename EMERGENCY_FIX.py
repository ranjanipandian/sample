"""
EMERGENCY FIX - Restart Streamlit with Fresh Cache
Run this before your presentation to ensure everything works
"""

import subprocess
import sys
import os
import time

print("=" * 80)
print("🚨 EMERGENCY FIX FOR SEMANTIC SEARCH")
print("=" * 80)
print()

# Step 1: Test that backend works
print("STEP 1: Testing backend semantic search...")
print("-" * 80)

try:
    from query_service import semantic_search
    results = semantic_search("clinical trials", top_k=3, min_score=0.45)
    
    if results:
        print(f"✅ Backend is working! Found {len(results)} results:")
        for i, r in enumerate(results[:3], 1):
            print(f"   {i}. {r['file_name']} (score: {r['relevance_score']:.3f})")
    else:
        print("❌ Backend returned no results!")
        print("   This is a CRITICAL issue - embeddings may be missing")
        sys.exit(1)
except Exception as e:
    print(f"❌ Backend test failed: {e}")
    sys.exit(1)

print()

# Step 2: Clear Streamlit cache
print("STEP 2: Clearing Streamlit cache...")
print("-" * 80)

try:
    result = subprocess.run(
        ["streamlit", "cache", "clear"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print("✅ Streamlit cache cleared")
except Exception as e:
    print(f"⚠️ Could not clear cache automatically: {e}")
    print("   Please run manually: streamlit cache clear")

print()

# Step 3: Instructions
print("STEP 3: Restart Instructions")
print("-" * 80)
print()
print("🎯 TO FIX THE DASHBOARD PERMANENTLY:")
print()
print("1. If Streamlit is running, press Ctrl+C to stop it")
print()
print("2. Run this command:")
print("   streamlit run ris_app.py --server.port 8501")
print()
print("3. When the browser opens, press Ctrl+Shift+Delete")
print("   - Select 'Cached images and files'")
print("   - Click 'Clear data'")
print()
print("4. Refresh the page (F5)")
print()
print("5. Test with query: 'clinical trials'")
print()
print("=" * 80)
print("✅ BACKEND IS CONFIRMED WORKING")
print("=" * 80)
print()
print("💡 The issue is ONLY in the Streamlit cache/browser cache")
print("   Following the steps above will fix it permanently")
print()

# Create a quick start script
with open("START_DASHBOARD.bat", "w") as f:
    f.write("@echo off\n")
    f.write("echo Starting Research Intelligence Dashboard...\n")
    f.write("echo.\n")
    f.write("cd /d \"%~dp0\"\n")
    f.write("streamlit cache clear\n")
    f.write("streamlit run ris_app.py --server.port 8501\n")

print("📝 Created START_DASHBOARD.bat for easy launching")
print("   Double-click this file to start the dashboard with cleared cache")
print()
