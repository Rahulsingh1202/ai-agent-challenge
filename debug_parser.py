import pandas as pd
import sys
sys.path.append('customparsers')
from iciciparser import parse

# Test paths
pdf_path = "data/icici/icicisample.pdf"
csv_path = "data/icici/icicisample.csv"

print("=== DEBUGGING PARSER OUTPUT ===")

# Load expected data
print("1. Loading expected CSV:")
df_expected = pd.read_csv(csv_path)
print(f"Expected shape: {df_expected.shape}")
print(f"Expected columns: {list(df_expected.columns)}")
print(f"Expected dtypes:\n{df_expected.dtypes}")
print("Expected first 3 rows:")
print(df_expected.head(3))
print()

# Test parser
print("2. Testing parser:")
try:
    df_parsed = parse(pdf_path)
    print(f"Parsed shape: {df_parsed.shape}")
    print(f"Parsed columns: {list(df_parsed.columns)}")
    print(f"Parsed dtypes:\n{df_parsed.dtypes}")
    print("Parsed first 3 rows:")
    print(df_parsed.head(3))
    print()
    
    # Detailed comparison
    print("3. Detailed Comparison:")
    print(f"Shapes match: {df_expected.shape == df_parsed.shape}")
    print(f"Columns match: {list(df_expected.columns) == list(df_parsed.columns)}")
    
    if df_expected.shape == df_parsed.shape and list(df_expected.columns) == list(df_parsed.columns):
        print("\n4. Row-by-row comparison:")
        for i in range(min(3, len(df_expected))):
            print(f"\nRow {i}:")
            print(f"Expected: {df_expected.iloc[i].tolist()}")
            print(f"Parsed:   {df_parsed.iloc[i].tolist()}")
            print(f"Match: {df_expected.iloc[i].equals(df_parsed.iloc[i])}")
    
    print(f"\n5. Overall equals: {df_expected.equals(df_parsed)}")
    
except Exception as e:
    print(f"Parser failed with error: {e}")
    import traceback
    traceback.print_exc()
