import os, logging, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, List, Any

from db import init_db, fetch_schema, safe_select
from llm_sql import build_chain, extract_sql

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
DB_PATH = os.getenv('DB_PATH','./data/app.db')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', "*").split(',')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Global variables
chain = None
schema = None

class GenerateSQLQuery(BaseModel):
    question: str = Field(..., 
        min_length=1, 
        max_length=1000, 
        description='Natural language question'
    )
    execute: bool = Field(
        default=False, 
        description='Whether to execute the generated SQL'
    )


class SQLResponse(BaseModel):
    sql: str
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global chain, schema

    try:
        # Initialize database and schema
        logger.info(f'Initializing database at {DB_PATH}')
        init_db(DB_PATH)
        schema = fetch_schema(DB_PATH)
        logger.info('Database initialized successfully!')

        logger.info('Building LLM chain')
        chain = build_chain()
        logger.info('LLM chain built successfully!')
    
    except Exception as e:
        logger.error(f'Failed to initialize application: {e}')
        raise

    yield

    logger.info('Application shutdown')

app = FastAPI(
    title='NL2SQL API',
    description='Natural Language to SQL query generation and execution service',
    version='1.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)

# Mount static files (frontend)
# app.mount("/", StaticFiles(directory='frontend', html=True), name='static')

@app.exception_handler(Exception)
async def global_execution_handler(request, exc):
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    return JSONResponse(
        status_code=500,
        content={'error': 'Internal server error'}
    )

def get_chain():
    """Dependency to get the LLM chain"""
    if chain is None:
        raise HTTPException(status_code=503, detail="Service not ready - LLM chain not initialized")
    return chain

def get_schema():
    """Dependency to get the database schema"""
    if schema is None:
        raise HTTPException(status_code=503, detail="Service not ready - Database schema not loaded")
    return schema

@app.get('/api/health')
def health_check():
    """Health check endpoint"""
    return {
        'status':'healthy',
        'database_initialized': schema is not None,
        'llm_chain_initialized': chain is not None
    }

@app.post('/api/nl2sql', response_model=SQLResponse)
def nl2sql(
    query: GenerateSQLQuery,
    llm_chain=Depends(get_chain),
    db_schema=Depends(get_schema)
):
    """Convert natural language question to SQL and optionally executes it"""
    try:
        logger.info(f"Processing question: {query.question}")

        # Generate SQL using LLM chain
        response = llm_chain.invoke({'schema':db_schema, 'question':query.question})

        sql = extract_sql(response)
        if not sql:
            raise HTTPException(status_code=400, detail='Could not generate valid SQL from question')
        
        logger.info(f'Generated SQL: {sql}')
        result = SQLResponse(sql=sql)

        # Execute SQL if requested
        if query.execute:
            try:
                columns, rows = safe_select(DB_PATH, sql)
                result.columns = columns
                result.rows = rows
                logger.info(f"SQL executed successfully, returned {len(rows)} rows")
            except Exception as e:
                error_msg = f"SQL execution error: {str(e)}"
                logger.error(error_msg)
                result.error = error_msg
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error processing request: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to process request')


if __name__ == '__main__':
    uvicorn.run('main:app', host=HOST, port=PORT, log_level=LOG_LEVEL.lower(), reload=True)
# --------------------------------------------------------------------------------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware, 
#     allow_origins=['*'], 
#     allow_methods=['*'], 
#     allow_headers=['*']
# )

# init_db(DB)
# SCHEMA = fetch_schema(DB)

# global chain
# chain = None

# class GenerateSQLQuery(BaseModel): 
#     question: str
#     execute: bool = False

# @app.get('/health')
# def checkHealth(): 
#     return {'ok': True }

# @app.post('/nl2sql')
# def nl2sql(p:GenerateSQLQuery):
#     chain = chain or build_chain()
#     out = chain.invoke({'schema':SCHEMA,'question':p.question})
#     sql = extract_sql(out)
#     print('\n[SQL]\n',sql)
#     res = {'sql':sql}
#     if p.execute:
#         try: 
#             cols, rows = safe_select(DB, sql)
#             res.update({'columns': cols, 'rows': rows})
#         except Exception as e: 
#             res['error'] = str(e)
#     return res

# if __name__=='__main__':
#     uvicorn.run('main:app',host='127.0.0.1',port=8000)
