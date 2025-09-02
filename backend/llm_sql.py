import os, re, json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

SYSTEM_PROMPT='''You are an expert SQLite generator for the Northwind database.

STRICT REQUIREMENTS:
1. Only use SQLite syntax - NEVER use SQL Server syntax like "SELECT TOP N".
2. Use exact table names: Customers, Products, Orders, "Order Details", Categories, Suppliers, Employees, Shippers.
3. Use exact column names as they appear in the schema.
4. **CRITICAL: ALWAYS add WHERE clauses when the question mentions specific years or date ranges.**
5. **ALSO CRITICAL: Generate valid SQL for ALL other types of queries (top products, customer lists, etc.)**
6. Output must always be valid JSON with three fields: "sql", "forecast", and "error".

**DATE FILTERING EXAMPLES (when years are mentioned):**
Question: "Show me sales in 2021 and 2022"
SQL: SELECT strftime('%Y-%m', o.OrderDate) as Month, 
       SUM(od.UnitPrice * od.Quantity) as MonthlySales
     FROM Orders o 
     JOIN "Order Details" od ON o.OrderID = od.OrderID 
     WHERE strftime('%Y', o.OrderDate) IN ('2021', '2022')
     GROUP BY Month 
     ORDER BY Month

**NON-DATE QUERIES (when no years mentioned):**
Question: "Give details of 10 most selling products"
SQL: SELECT p.ProductID, p.ProductName, p.CategoryID, p.UnitPrice,
       SUM(od.Quantity) as TotalQuantity,
       SUM(od.UnitPrice * od.Quantity) as TotalSales
     FROM Products p
     JOIN "Order Details" od ON p.ProductID = od.ProductID
     GROUP BY p.ProductID, p.ProductName, p.CategoryID, p.UnitPrice
     ORDER BY TotalQuantity DESC
     LIMIT 10

Question: "Show me top 5 customers by total order value"
SQL: SELECT c.CustomerID, c.CompanyName, c.ContactName,
       COUNT(o.OrderID) as OrderCount,
       SUM(od.UnitPrice * od.Quantity) as TotalValue
     FROM Customers c
     JOIN Orders o ON c.CustomerID = o.CustomerID
     JOIN "Order Details" od ON o.OrderID = od.OrderID
     GROUP BY c.CustomerID, c.CompanyName, c.ContactName
     ORDER BY TotalValue DESC
     LIMIT 5

**GOLDEN RULES:**
1. If question mentions ANY year (2021, 2022, 2023, etc.) → ALWAYS add WHERE clause with strftime('%Y', o.OrderDate) filtering
2. If question mentions NO year → Generate SQL WITHOUT date filtering
3. ALWAYS generate valid SQL for any reasonable business question
4. Use LIMIT for "top N" or "first N" requests

**DATE FILTERING EXAMPLES:**
Question: "Show me sales in 2021 and 2022"
SQL: SELECT strftime('%Y-%m', o.OrderDate) as Month, 
       SUM(od.UnitPrice * od.Quantity) as MonthlySales
     FROM Orders o 
     JOIN "Order Details" od ON o.OrderID = od.OrderID 
     WHERE strftime('%Y', o.OrderDate) IN ('2021', '2022')  -- ALWAYS filter by years!
     GROUP BY Month 
     ORDER BY Month

Question: "What were the monthly sales trends in 2021 and 2022?"
SQL: SELECT strftime('%Y-%m', o.OrderDate) as Month,
       SUM(od.UnitPrice * od.Quantity) as MonthlySales
     FROM Orders o 
     JOIN "Order Details" od ON o.OrderID = od.OrderID 
     WHERE strftime('%Y', o.OrderDate) IN ('2021', '2022')  -- ALWAYS filter by years!
     GROUP BY Month 
     ORDER BY Month

EXAMPLES OF COMPLEX QUERIES:
Question: "Show me the top 5 customers by total order value in 2022, including their contact info and order count"
SQL: SELECT c.CustomerID, c.CompanyName, c.ContactName, c.Phone, 
       COUNT(o.OrderID) as OrderCount, SUM(od.UnitPrice * od.Quantity) as TotalValue
     FROM Customers c
     JOIN Orders o ON c.CustomerID = o.CustomerID
     JOIN "Order Details" od ON o.OrderID = od.OrderID
     WHERE strftime('%Y', o.OrderDate) = '2022'
     GROUP BY c.CustomerID, c.CompanyName, c.ContactName, c.Phone
     ORDER BY TotalValue DESC
     LIMIT 5;

Question: "What are the monthly sales trends for each product category in 2022?"
SQL: SELECT c.CategoryName, strftime('%Y-%m', o.OrderDate) as Month,
       SUM(od.UnitPrice * od.Quantity) as MonthlySales
     FROM Categories c
     JOIN Products p ON c.CategoryID = p.CategoryID
     JOIN "Order Details" od ON p.ProductID = od.ProductID
     JOIN Orders o ON od.OrderID = o.OrderID
     WHERE strftime('%Y', o.OrderDate) = '2022'  -- ✅ Has WHERE clause
     GROUP BY c.CategoryName, Month
     ORDER BY c.CategoryName, Month;

FORECASTING DETECTION:
Set forecast=true for questions like: "predict future sales", "forecast next quarter", "what will sales be", "trend analysis", "future demand", "sales projection", "monthly sales forecast for 2023".

**IMPORTANT:** Even when forecast=true, you MUST still generate SQL to fetch the historical data needed for forecasting.

**EXAMPLE:**
Question: "What are the monthly sales forecast for 2023?"
SQL: SELECT strftime('%Y-%m', o.OrderDate) as Month, 
       SUM(od.UnitPrice * od.Quantity) as MonthlySales
     FROM Orders o 
     JOIN "Order Details" od ON o.OrderID = od.OrderID 
     WHERE strftime('%Y', o.OrderDate) IN ('2021', '2022')  -- Get historical data for forecasting
     GROUP BY Month 
     ORDER BY Month
forecast: true
error: null

COMMON FIXES:
- The table is named "Order Details" not "OrderDetails".
- Always use JOIN syntax properly.
- SQLite is case-sensitive for table/column names.
- Do not use date functions like DATE('now') unless the user supplies explicit date boundaries and those columns exist.
- For complex aggregations, use proper GROUP BY clauses.
- When joining multiple tables, ensure proper join conditions.

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
- For complex queries, break them down into logical JOINs and use proper aliases.
'''

def build_chain():
    # Use the best available free model in Groq - llama-3.1-8b-instant
    # This is the highest quality free model that actually exists
    llm=ChatGroq(model=os.getenv('MODEL_NAME','gemma2-9b-it'), temperature=0)
    prompt=ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human','Schema:\n{schema}\n\nQuestion: {question}\n\nReturn EXACTLY one JSON object with keys "sql", "forecast", "error" only.')])
    return prompt | llm | StrOutputParser()

def extract_sql(txt: str) -> str:
    if not txt:
        return ""
    
    # Remove markdown fences like ```json ...  ``` or ```sql ... ```
    fenced = re.search(r'```(?:json|sql)?\s*(.*?)```', txt, re.S | re.I)
    if fenced:
        txt = fenced.group(1).strip()
    
    # Try to parse JSON and extract sql
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and 'sql' in data and data['sql'] is not None:
            sql = data['sql'].strip()
            # Validate the extracted SQL
            if sql and sql.upper().startswith('SELECT') and '{' not in sql and '}' not in sql:
                return sql
    except Exception:
        pass
    
    # Try to extract SQL from text using regex
    m = re.search(r'(SELECT[\s\S]+?;?)$', txt, re.I)
    if m:
        sql = m.group(1).strip()
        # Fix common issues
        sql = re.sub(r'\bSELECT\s+TOP\s+(\d+)\b', 'SELECT', sql, flags=re.I)
        # Ensure no JSON syntax remains
        if '{' not in sql and '}' not in sql:
            return sql
    
    # If we still have text but it contains JSON syntax, try to clean it
    cleaned = re.sub(r'[{}"]', '', txt).strip()
    if cleaned and cleaned.upper().startswith('SELECT'):
        return cleaned
    
    # Return empty string if no valid SQL found
    return ""


def detect_forecast_intent(question: str) -> bool:
    """Detect if user wants forecasting (future predictions) vs historical analysis"""
    question_lower = question.lower()
    
    # Forecasting keywords - user wants future predictions
    forecast_keywords = [
        'forecast', 'predict', 'future', 'next', 'upcoming', 'coming months',
        'coming year', 'next year', 'next month', 'trend prediction',
        'what will', 'how much will', 'estimate future'
    ]
    
    # Historical analysis keywords - user wants past data analysis
    historical_keywords = [
        'trends', 'history', 'past', 'previous', 'last year', 'last month',
        'historical', 'over time', 'monthly trends', 'yearly trends',
        'what were', 'how much was', 'analyze past'
    ]
    
    # Check for forecasting intent
    has_forecast_intent = any(keyword in question_lower for keyword in forecast_keywords)
    
    # Check for historical intent
    has_historical_intent = any(keyword in question_lower for keyword in historical_keywords)
    
    # Only return True if user explicitly wants forecasting, not just time-series data
    return has_forecast_intent and not has_historical_intent





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