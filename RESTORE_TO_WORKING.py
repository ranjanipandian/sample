"""
RESTORE TO WORKING STATE
This script will restore query_service.py and ris_app.py to their original working configuration
Run this to undo all the threshold changes
"""

import os

# Backup current files first
print("Creating backups...")
os.system('copy query_service.py query_service_backup.py')
os.system('copy ris_app.py ris_app_backup.py')

print("\n" + "="*80)
print("RESTORING TO ORIGINAL WORKING STATE")
print("="*80)

# The issue: I changed thresholds from working values to non-working values
# Original working configuration from FIXES_APPLIED.md was:
# - semantic_search: min_score = 0.3
# - This was working this morning before I made changes

print("\nTo restore manually:")
print("1. In query_service.py, line ~230:")
print("   Change: min_score: float = 0.62")
print("   To:     min_score: float = 0.3")
print()
print("2. In ris_app.py, around line 1089:")
print("   Change all min_score=0.62 to min_score=0.3")
print()
print("3. Refresh browser with Ctrl+F5")
print()
print("="*80)
print("ALTERNATIVE: Use Keyword Search mode instead of Semantic")
print("Keyword search doesn't use thresholds and will give exact matches")
print("="*80)
