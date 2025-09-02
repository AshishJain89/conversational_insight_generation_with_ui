import os, re, json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

SYSTEM_PROMPT='''You are an expert SQLite generator for the Northwind database.

STRICT REQUIREMENTS:
1. Only use SQLite syntax - NEVER use SQL Server syntax like "SELECT TOP N".
2. Use exact table names: Customers, Products, Orders, "Order Details", Categories, Suppliers, Employees, Shippers.
3. Use exact column names as they appear in the schema.
4. Output must always be valid JSON with three fields: "sql", "forecast", and "error".
   - "sql": a single valid SELECT query string for SQLite, or null if none is produced.
   - "forecast": boolean, true if the user asks for forecasting.
   - "error": null if no error; otherwise a short machine-readable message.

FORECASTING DETECTION:
Set forecast=true for questions like: "predict future sales", "forecast next quarter", "what will sales be", "trend analysis", "future demand", "sales projection".
Set forecast=false for historical/current data queries like: "show sales", "top products", "customer list".

COMMON FIXES:
- The table is named "Order Details" not "OrderDetails".
- Always use JOIN syntax properly.
- SQLite is case-sensitive for table/column names.
- Do not use date functions like DATE('now') unless the user supplies explicit date boundaries and those columns exist.

RESPONSE FORMAT (JSON only):
Return exactly one JSON object and nothing else. Examples (literal braces are escaped here to prevent template parsing):

Success example:
{{"sql":"SELECT CustomerID, CompanyName FROM Customers ORDER BY CustomerID LIMIT 10;","forecast":false,"error":null}}

Forecast intent example:
{{"sql":null,"forecast":true,"error":null}}

Missing column example:
{{"sql":null,"forecast":false,"error":"Missing column: OrderDetails.UnitsSold"}}

RULES:
- Parse the provided Schema text to build exact table/column names.
- If a requested table/column is not in Schema, do not guess: return error and sql=null.
- Always return a single SELECT statement when applicable.
- Never output explanations, markdown, or any text outside the single JSON object.
'''

def build_chain():
    llm=ChatGroq(model=os.getenv('MODEL_NAME','llama-3.1-8b-instant'), temperature=0)
    prompt=ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human','Schema:\n{schema}\n\nQuestion: {question}\n\nReturn EXACTLY one JSON object with keys "sql", "forecast", "error" only.')])
    return prompt | llm | StrOutputParser()

def extract_sql(txt: str) -> str:
    # Remove markdown fences like ```json ...  ``` or ```sql ... ```
    fenced = re.search(r'```(?:json|sql)?\s*(.*?)```', txt, re.S | re.I)
    if fenced:
        txt = fenced.group(1).strip()
    
    # Try to parse JSON and extract sql
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and 'sql' in data and data['sql'] is not None:
            return data['sql'].strip()
    except Exception:
        pass
    
    # Try to extract SQL from text
    m = re.search(r'(SELECT[\s\S]+?;?)$', txt, re.I)
    if m:
        sql = m.group(1).strip()
        # Fix common issues
        sql = re.sub(r'\bSELECT\s+TOP\s+(\d+)\b', 'SELECT', sql, flags=re.I)
        return sql
    
    # Otherwise return cleaned string
    return txt.strip()


def detect_forecast_intent(txt: str) -> bool:
    keywords = [
        'forecast', 'predict', 'projection', 'future', 'trend', 'next quarter', 'next year', 'what will'
    ]
    t = txt.lower()
    return any(k in t for k in keywords)





# import os, re
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq

# SYSTEM_PROMPT='''You are an expert SQLite generator for the Northwind database. 

# STRICT REQUIREMENTS:
# 1. Only use SQLite syntax - NEVER use SQL Server syntax like "SELECT TOP N"
# 2. Always use "LIMIT N" at the end of queries, never "TOP"
# 3. Use exact table names: Customers, Products, Orders, OrderDetails, Categories, Suppliers, Employees, Shippers
# 4. Use exact column names as they appear in the schema
# 5. Output only a single SELECT query that works in SQLite

# COMMON FIXES:
# - "OrderDetails" not "Order_Details" 
# - Use proper JOIN syntax
# - SQLite is case-sensitive for table/column names'''

# def build_chain():
#     llm=ChatGroq(model=os.getenv('MODEL_NAME','llama-3.1-8b-instant'), temperature=0)
#     prompt=ChatPromptTemplate.from_messages([
#         ('system', SYSTEM_PROMPT),
#         ('human','Schema:\n{schema}\n\nQuestion: {question}\n\nGenerate SQLite-compatible SQL only in ```sql``` block')])
#     return prompt | llm | StrOutputParser()

# def extract_sql(txt:str)->str:
#     m=re.search(r'```sql\s*(.*?)```',txt,re.S|re.I)
#     sql = m.group(1).strip() if m else txt.strip()
    
#     # Fallback fixes for common issues
#     if 'TOP ' in sql.upper():
#         sql = re.sub(r'\bSELECT\s+TOP\s+(\d+)\b', r'SELECT', sql, flags=re.IGNORECASE)
#         if 'LIMIT' not in sql.upper():
#             sql = sql.rstrip(';') + ' LIMIT 10;'
    
#     return sql