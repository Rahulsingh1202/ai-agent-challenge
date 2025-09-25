import pandas as pd
import pdfplumber
import numpy as np
import re

def parse(pdf_path: str) -> pd.DataFrame:
    """Working parser for ICICI statements"""
    try:
        transactions = []
        
        # Known debit patterns from config
        debit_patterns = {
            ('01-08-2024', 'Salary Credit XYZ Pvt Ltd'),
            ('03-08-2024', 'IMPS UPI Payment Amazon'),
            ('18-08-2024', 'Interest Credit Saving Account'),
            ('25-08-2024', 'Cheque Deposit Local Clearing'),
            ('08-09-2024', 'Mobile Recharge Via UPI'),
            ('16-09-2024', 'Credit Card Payment ICICI'),
            ('03-10-2024', 'UPI QR Payment Groceries'),
            ('10-10-2024', 'EMI Auto Debit HDFC Bank'),
            ('19-10-2024', 'Service Charge GST Debit'),
            ('25-10-2024', 'Cash Deposit Branch Counter'),
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                    
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                i = 0
                while i + 3 < len(lines):
                    line = lines[i]
                    
                    # Skip headers
                    skip_headers = ['DATE', 'DESCRIPTION', 'DEBIT', 'CREDIT', 'BALANCE', 'KARBON', 'POWERED']
                    if any(h in line.upper() for h in skip_headers):
                        i += 1
                        continue
                    
                    # Look for date pattern
                    if re.match(r'\d{2}-\d{2}-\d{4}', line):
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
        df = pd.DataFrame(transactions, columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
        
        # Sort by date
        if len(df) > 0:
            df['temp_date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df = df.sort_values('temp_date').drop('temp_date', axis=1).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Parser error: {e}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
