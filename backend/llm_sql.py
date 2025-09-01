import os, re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

SYSTEM_PROMPT='''You are an expert SQLite generator. Only output a single SELECT query.'''

def build_chain():
    llm=ChatGroq(model=os.getenv('MODEL_NAME','llama-3.1-8b-instant'), temperature=0)
    prompt=ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        ('human','Schema:\n{schema}\nQuestion: {question}\nSQL only in ```sql``` block')])
    return prompt|llm|StrOutputParser()

def extract_sql(txt:str)->str:
    m=re.search(r'```sql\s*(.*?)```',txt,re.S|re.I)
    return m.group(1).strip() if m else txt.strip()
