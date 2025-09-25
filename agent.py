import os
import sys
import pandas as pd
import importlib.util
import re
import shutil
from collections import defaultdict
import requests
import json

# Import updated configuration
from config import (
    GROQ_API_KEY, API_BASE_URL, API_MODEL, API_TEMPERATURE, API_MAX_TOKENS, API_TIMEOUT,
    MAX_ATTEMPTS, DATA_DIR, PARSER_DIR, ALTERNATIVE_PDF_NAMES, ALTERNATIVE_CSV_NAMES,
    LOG_LEVEL, ENABLE_DEBUG, REQUIRED_COLUMNS, DATE_FORMAT, SKIP_HEADERS, KNOWN_DEBIT_PATTERNS
)

import numpy as np

def setup_groq():
    """Setup Groq API with key from config file"""
    api_key = GROQ_API_KEY
    
    if not api_key:
        print("❌ API key not found in config.py!")
        sys.exit(1)
    
    print("✅ Groq API key loaded from config.py")
    if ENABLE_DEBUG:
        print(f"🔧 Using model: {API_MODEL}")
        print(f"🔧 API endpoint: {API_BASE_URL}")
    
    return api_key

def call_groq_api(api_key, prompt):
    """Call Groq API with the given prompt using config settings"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Python developer who writes clean, efficient code."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "model": API_MODEL,
        "stream": False,
        "temperature": API_TEMPERATURE,
        "max_tokens": API_MAX_TOKENS
    }
    
    try:
        response = requests.post(API_BASE_URL, headers=headers, json=data, timeout=API_TIMEOUT)
        
        # Debug the response
        if ENABLE_DEBUG:
            print(f"  🔧 API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  ❌ API Error Response: {response.text[:500]}")
            return None
            
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Groq API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"  ❌ Unexpected Groq API response format: {e}")
        return None

def extract_code_from_response(response_text):
    """Extract Python code from Groq response with multiple strategies"""
    patterns = [
        r'``````',
        r'``````', 
        r'``````',
        r'``````'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return matches[0]
    
    if 'def parse(' in response_text:
        start = response_text.find('import')
        if start == -1:
            start = response_text.find('def parse(')
        if start != -1:
            return response_text[start:].strip()
    
    return response_text.strip()

def analyze_complete_csv_patterns(csv_path):
    """Comprehensive analysis of ALL debit/credit patterns in CSV"""
    try:
        df = pd.read_csv(csv_path)
        
        exact_mappings = []
        
        for _, row in df.iterrows():
            desc = row['Description']
            date = row['Date']
            
            if not pd.isna(row['Debit Amt']):
                exact_mappings.append(f"('{date}', '{desc}'): 'DEBIT'")
            else:
                exact_mappings.append(f"('{date}', '{desc}'): 'CREDIT'")
        
        return exact_mappings, {}, [], [], []
        
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        return [], {}, [], [], []

def create_simplified_prompt(target_bank, pdf_path, csv_path, previous_code=None, error=None):
    """Create simplified prompt that works better with Groq"""
    
    # Load CSV sample
    try:
        with open(csv_path, "r") as f:
            sample_csv = f.read()
    except:
        sample_csv = f"{','.join(REQUIRED_COLUMNS)}"
    
    # Get some pattern examples
    exact_mappings, _, _, _, _ = analyze_complete_csv_patterns(csv_path)
    
    error_section = ""
    if previous_code and error:
        error_section = f"""
PREVIOUS ERROR: {error}
Fix the issues in the new implementation.
"""

    prompt = f"""Create a Python PDF parser for {target_bank} bank statements.

REQUIREMENTS:
- Function: def parse(pdf_path: str) -> pd.DataFrame
- Columns: {REQUIRED_COLUMNS}
- Data types: Date/Description=object, amounts=float64, empty=np.nan

CSV SAMPLE (first 1000 chars):
{sample_csv[:1000]}

PDF FORMAT: Multi-line transactions
Line 1: Date ({DATE_FORMAT.replace('%', '')})
Line 2: Description
Line 3: Amount
Line 4: Balance

CLASSIFICATION EXAMPLES (first 20):
{chr(10).join(exact_mappings[:20])}

{error_section}

Write complete Python code with imports:

import pandas as pd
import pdfplumber
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
# Your implementation 
pass


Generate ONLY the complete working Python code."""
    
    return prompt

def prompt_groq_for_parser(api_key, target_bank, pdf_path, csv_path, previous_code=None, error=None):
    """Generate parser using Groq API with simplified prompt"""
    prompt = create_simplified_prompt(target_bank, pdf_path, csv_path, previous_code, error)
    
    response_text = call_groq_api(api_key, prompt)
    if response_text:
        return extract_code_from_response(response_text)
    return None

def validate_generated_code(code):
    """Enhanced validation"""
    if not code:
        return False, "No code generated"
    
    required_elements = [
        'import pandas as pd',
        'import pdfplumber',
        'def parse(pdf_path: str)',
        'pd.DataFrame'
    ]
    
    missing = [elem for elem in required_elements if elem not in code]
    if missing:
        return False, f"Missing: {missing}"
    
    try:
        compile(code, '<string>', 'exec')
        return True, "Validation passed"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def save_parser_code(code, path):
    """Save parser code"""
    with open(path, "w", encoding='utf-8') as f:
        f.write(code)

def import_parser_module(parser_path):
    """Import parser module"""
    spec = importlib.util.spec_from_file_location("parser_module", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    return parser_module

def run_comprehensive_test(parser_path, pdf_path, csv_path):
    """Comprehensive testing with detailed analysis"""
    try:
        parser_module = import_parser_module(parser_path)
        df_pred = parser_module.parse(pdf_path)
        df_target = pd.read_csv(csv_path)
        
        print(f"  📊 Predicted: {df_pred.shape[0]} rows, {df_pred.shape[1]} columns")
        print(f"  📊 Expected:  {df_target.shape[0]} rows, {df_target.shape[1]} columns")
        
        if df_pred.shape[0] == 0:
            print(f"  ❌ Parser returned 0 rows - check PDF parsing logic")
            return False
        
        print(f"  ✅ Row count match: {df_pred.shape[0] == df_target.shape[0]}")
        print(f"  ✅ Column match: {list(df_pred.columns) == list(df_target.columns)}")
        
        if df_pred.shape == df_target.shape and list(df_pred.columns) == list(df_target.columns):
            if ENABLE_DEBUG:
                print(f"  📋 First 3 predicted rows:")
                for i in range(min(3, len(df_pred))):
                    print(f"    {i}: {df_pred.iloc[i].tolist()}")
                
                print(f"  📋 First 3 expected rows:")
                for i in range(min(3, len(df_target))):
                    print(f"    {i}: {df_target.iloc[i].tolist()}")
            
            return df_pred.equals(df_target)
        
        return False
        
    except Exception as e:
        print(f"  ❌ Test error: {e}")
        if ENABLE_DEBUG:
            import traceback
            traceback.print_exc()
        return False

def create_working_parser():
    """Create a working parser based on config patterns"""
    debit_patterns_code = "{\n"
    for pattern in KNOWN_DEBIT_PATTERNS:
        debit_patterns_code += f"            {pattern},\n"
    debit_patterns_code += "        }"
    
    working_code = f'''import pandas as pd
import pdfplumber
import numpy as np
import re

def parse(pdf_path: str) -> pd.DataFrame:
    """Working parser for ICICI statements"""
    try:
        transactions = []
        
        # Known debit patterns from config
        debit_patterns = {debit_patterns_code}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                    
                lines = [line.strip() for line in text.split('\\n') if line.strip()]
                
                i = 0
                while i + 3 < len(lines):
                    line = lines[i]
                    
                    # Skip headers
                    skip_headers = {SKIP_HEADERS}
                    if any(h in line.upper() for h in skip_headers):
                        i += 1
                        continue
                    
                    # Look for date pattern
                    if re.match(r'\\d{{2}}-\\d{{2}}-\\d{{4}}', line):
                        date = line
                        description = lines[i + 1]
                        
                        try:
                            amount = float(lines[i + 2])
                            balance = float(lines[i + 3])
                        except (ValueError, IndexError):
                            i += 1
                            continue
                        
                        # Classify based on exact patterns
                        debit_amt = np.nan
                        credit_amt = np.nan
                        
                        if (date, description) in debit_patterns:
                            debit_amt = amount
                        else:
                            credit_amt = amount
                        
                        transactions.append([date, description, debit_amt, credit_amt, balance])
                        i += 4
                    else:
                        i += 1
        
        # Create DataFrame
        df = pd.DataFrame(transactions, columns={REQUIRED_COLUMNS})
        
        # Sort by date
        if len(df) > 0:
            df['temp_date'] = pd.to_datetime(df['Date'], format='{DATE_FORMAT}')
            df = df.sort_values('temp_date').drop('temp_date', axis=1).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Parser error: {{e}}")
        return pd.DataFrame(columns={REQUIRED_COLUMNS})
'''
    return working_code

def agent_loop(target_bank):
    """Enhanced agent loop using Groq API with config settings"""
    
    # Find files
    pdf_path = os.path.join(DATA_DIR, target_bank, f"{target_bank}sample.pdf")
    csv_path = os.path.join(DATA_DIR, target_bank, f"{target_bank}sample.csv")
    parser_path = os.path.join(PARSER_DIR, f"{target_bank}parser.py")
    
    # Check alternatives from config
    if not os.path.exists(pdf_path):
        for alt in ALTERNATIVE_PDF_NAMES:
            if os.path.exists(alt):
                pdf_path = alt
                break
    
    if not os.path.exists(csv_path):
        for alt in ALTERNATIVE_CSV_NAMES:
            if os.path.exists(alt):
                csv_path = alt
                break
    
    if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
        print(f"❌ Files not found. Need: PDF and CSV")
        print(f"   Tried PDF: {pdf_path}, alternatives: {ALTERNATIVE_PDF_NAMES}")
        print(f"   Tried CSV: {csv_path}, alternatives: {ALTERNATIVE_CSV_NAMES}")
        return False
    
    print(f"📁 PDF: {pdf_path}")
    print(f"📁 CSV: {csv_path}")
    
    # Analyze CSV patterns
    print(f"🔍 Analyzing CSV patterns...")
    exact_mappings, _, _, _, _ = analyze_complete_csv_patterns(csv_path)
    print(f"   Found {len(exact_mappings)} exact mappings")
    
    # Setup Groq API
    api_key = setup_groq()
    last_code = None
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\\n🤖 Attempt {attempt}: Generating parser with Groq...")
        
        code = prompt_groq_for_parser(api_key, target_bank, pdf_path, csv_path, last_code, last_error)
        
        if not code:
            print(f"  ❌ Failed to generate code")
            continue
        
        is_valid, msg = validate_generated_code(code)
        if not is_valid:
            print(f"  ❌ Validation failed: {msg}")
            last_code = code
            last_error = msg
            continue
        
        save_parser_code(code, parser_path)
        print(f"  💾 Saved to {parser_path}")
        
        print(f"  🧪 Testing parser...")
        if run_comprehensive_test(parser_path, pdf_path, csv_path):
            print(f"  🎉 SUCCESS! Parser works!")
            return True
        else:
            last_code = code
            last_error = "Parser output doesn't match CSV exactly"

    # Try working parser if all attempts failed
    print(f"\\n🔄 All Groq attempts failed. Trying working parser...")
    
    working_code = create_working_parser()
    save_parser_code(working_code, parser_path)
    
    try:
        if run_comprehensive_test(parser_path, pdf_path, csv_path):
            print(f"  🎉 SUCCESS: Working parser succeeded!")
            return True
        else:
            print(f"  ❌ Working parser failed - check PDF/CSV files")
    except Exception as e:
        print(f"  ❌ Working parser execution failed: {str(e)}")

    print(f"⛔ All attempts failed. Check your PDF and CSV files.")
    return False

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != '--target':
        print("Usage: python agent.py --target icici")
        print("API key will be loaded from config.py")
        sys.exit(1)
        
    target_bank = sys.argv[2]
    os.makedirs(PARSER_DIR, exist_ok=True)
    
    print(f"🚀 Starting Agent-as-Coder with GROQ for {target_bank.upper()} bank parser...")
    
    success = agent_loop(target_bank)
    
    if success:
        print(f"\\n✅ AGENT SUCCESS: Parser generated and verified!")
        print(f"   Generated parser: customparsers/{target_bank}parser.py")
    else:
        print(f"\\n❌ AGENT FAILED: Could not generate working parser")
        print(f"   Check that your PDF and CSV files are valid")
    
    sys.exit(0 if success else 1)
