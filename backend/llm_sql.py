import os, re, json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

SYSTEM_PROMPT='''You are an expert SQLite generator for the Northwind database. 

STRICT REQUIREMENTS:
1. Only use SQLite syntax - NEVER use SQL Server syntax like "SELECT TOP N"
2. Use exact table names: Customers, Products, Orders, Order Details, Categories, Suppliers, Employees, Shippers
3. Use exact column names as they appear in the schema
4. Output only a single SELECT query that works in SQLite

RESPONSE FORMAT:
Return JSON with two fields:
- "sql": the SQLite query
- "forecast": boolean - true if user asks for predictions, forecasting, future trends, "what will happen", "predict sales", "forecast demand", etc.

FORECASTING DETECTION:
Set forecast=true for questions like: "predict future sales", "forecast next quarter", "what will sales be", "trend analysis", "future demand", "sales projection"
Set forecast=false for historical/current data queries like: "show sales", "top products", "customer list"

COMMON FIXES:
- There is no table named "OrderDetails", check the given list of tables.
- Use proper JOIN syntax
- SQLite is case-sensitive for table/column names'''

def build_chain():
    llm=ChatGroq(model=os.getenv('MODEL_NAME','llama-3.1-8b-instant'), temperature=0)
    prompt=ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human','Schema:\n{schema}\n\nQuestion: {question}\n\nGenerate SQLite-compatible SQL only in ```sql``` block')])
    return prompt|llm|StrOutputParser()

def extract_sql(txt: str) -> str:
    # Remove markdown fences like ```json ...  ``` or ```sql ... ```
    fenced = re.search(r'```(?:json|sql)?\s*(.*?)```', txt, re.S | re.I)
    if fenced:
        txt = fenced.group(1).strip()
    
    # Try to parse JSON and extract sql
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and 'sql' in data:
            return data['sql']
    except Exception:
        pass

    # Otherwise return cleaned string
    return txt.strip()

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