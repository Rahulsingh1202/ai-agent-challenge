import pdfplumber
import re

pdf_path = "data/icici/icicisample.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    
    for i, page in enumerate(pdf.pages):
        print(f"\n=== Page {i+1} ===")
        text = page.extract_text()
        lines = text.split('\n') if text else []
        
        print(f"Total lines: {len(lines)}")
        
        # Show lines that contain dates
        date_lines = [line for line in lines if re.search(r'\d{2}-\d{2}-\d{4}', line)]
        print(f"Date-containing lines: {len(date_lines)}")
        
        # Show first few date lines
        for j, line in enumerate(date_lines[:5]):
            print(f"  {j+1}: {line}")
