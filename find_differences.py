import pandas as pd
import sys
sys.path.append('customparsers')
from iciciparser import parse

# Test paths
pdf_path = "data/icici/icicisample.pdf"
csv_path = "data/icici/icicisample.csv"

# Load data
df_expected = pd.read_csv(csv_path)
df_parsed = parse(pdf_path)

print("=== FINDING EXACT DIFFERENCES ===\n")

if df_expected.shape != df_parsed.shape:
    print(f"Shape mismatch: Expected {df_expected.shape}, Got {df_parsed.shape}")
else:
    print(f"Shapes match: {df_expected.shape} ✅")

print(f"Columns match: {list(df_expected.columns) == list(df_parsed.columns)} ✅")

# Find rows that don't match
mismatches = []
for i in range(len(df_expected)):
    if not df_expected.iloc[i].equals(df_parsed.iloc[i]):
        mismatches.append(i)

print(f"\nTotal mismatching rows: {len(mismatches)}")

# Show first 5 mismatches
for i, row_idx in enumerate(mismatches[:5]):
    print(f"\n=== MISMATCH {i+1}: Row {row_idx} ===")
    print(f"Expected: {df_expected.iloc[row_idx].tolist()}")
    print(f"Parsed:   {df_parsed.iloc[row_idx].tolist()}")
    
    # Check each column
    for col in df_expected.columns:
        exp_val = df_expected.iloc[row_idx][col]
        parsed_val = df_parsed.iloc[row_idx][col]
        if pd.isna(exp_val) and pd.isna(parsed_val):
            continue  # Both NaN
        elif exp_val != parsed_val:
            print(f"  Difference in '{col}': Expected '{exp_val}' != Parsed '{parsed_val}'")

if len(mismatches) == 0:
    print("🎉 ALL ROWS MATCH! Something else might be causing df.equals() to fail.")
    
    # Check for dtype issues
    print("\nChecking dtypes:")
    for col in df_expected.columns:
        if df_expected[col].dtype != df_parsed[col].dtype:
            print(f"  Dtype mismatch in '{col}': Expected {df_expected[col].dtype} != Parsed {df_parsed[col].dtype}")
