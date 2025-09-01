from prettytable import PrettyTable
import os, logging, uvicorn, sqlite3
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, List, Any

from .db import fetch_schema, safe_select
from .llm_sql import build_chain, extract_sql

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DB_PATH = os.getenv('DB_PATH','./data/northwind.db')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8080))
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

chain = None
schema = None

class GenerateSQLQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description='Natural language question')
    execute: bool = Field(default=True, description='Whether to execute the generated SQL')

class SQLResponse(BaseModel):
    sql: str
    table: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, schema
    try:
        logger.info(f'Initializing database at {DB_PATH}')
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

app = FastAPI(title='NL2SQL API', description='Natural Language to SQL query generation and execution service', version='1.0.0', lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=['GET','POST'], allow_headers=['*'])

@app.exception_handler(Exception)
async def global_execution_handler(request, exc):
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    return JSONResponse(status_code=500, content={'error': 'Internal server error'})

def get_chain():
    if chain is None:
        raise HTTPException(status_code=503, detail="Service not ready - LLM chain not initialized")
    return chain

def get_schema():
    if schema is None:
        raise HTTPException(status_code=503, detail="Service not ready - Database schema not loaded")
    return schema

@app.get('/api/health')
def health_check():
    return {'status':'healthy','database_initialized': schema is not None,'llm_chain_initialized': chain is not None}

@app.get('/api/test')
def test_data():
    cols, rows = safe_select('SELECT * FROM customers', DB_PATH)
    return {'columns': cols, 'rows': rows}

@app.post('/api/nl2sql', response_model=SQLResponse)
def nl2sql(query: GenerateSQLQuery, llm_chain=Depends(get_chain), db_schema=Depends(get_schema)):
    try:
        logger.info(f"Processing question: {query.question}")
        response = llm_chain.invoke({'schema': db_schema, 'question': query.question})

        # Parse response
        response_dict = {}
        try:
            response_dict = eval(response) if response.strip().startswith('{') else {}
        except Exception:
            response_dict = {}

        sql = response_dict.get('sql', extract_sql(response))
        forecast = response_dict.get('forecast', False)

        result = SQLResponse(sql=sql)

        if query.execute:
            if forecast:
                result.message = 'Forecast detected; SQL not executed.'
            else:
                try:
                    columns, rows = safe_select(sql, DB_PATH)
                    table = PrettyTable() if rows else None
                    if table:
                        table.field_names = columns
                        for row in rows:
                            table.add_row(row)
                        result.table = str(table)
                    else:
                        result.table = 'No data found'
                    result.columns = columns
                    result.rows = rows
                    logger.info(f"SQL executed successfully, returned {len(rows)} rows")
                except Exception as e:
                    result.message = f"SQL execution error: {e}"
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error processing request: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to process request')

@app.get('/')
def read_root():
    return FileResponse('frontend/dist/index.html')

app.mount('/static', StaticFiles(directory='frontend/dist', html=True), name='static')

if __name__ == '__main__':
    uvicorn.run('main:app', host=HOST, port=PORT, log_level=LOG_LEVEL.lower(), reload=True)

# from prettytable import PrettyTable
# import os, logging, uvicorn, sqlite3
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# from typing import Optional, List, Any

# from .db import fetch_schema, safe_select
# from .llm_sql import build_chain, extract_sql

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# load_dotenv()

# # Configuration
# DB_PATH = os.getenv('DB_PATH','./data/northwind.db')
# HOST = os.getenv('HOST', '0.0.0.0')
# PORT = int(os.getenv('PORT', 8080))
# ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', "*").split(',')
# LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# # Global variables
# chain = None
# schema = None

# class GenerateSQLQuery(BaseModel):
#     question: str = Field(..., 
#         min_length=1, 
#         max_length=1000, 
#         description='Natural language question'
#     )
#     execute: bool = Field(
#         default=True, 
#         description='Whether to execute the generated SQL'
#     )


# class SQLResponse(BaseModel):
#     sql: str
#     table: Optional[str] = None
#     columns: Optional[List[str]] = None
#     rows: Optional[List[List[Any]]] = None
#     message: Optional[str] = None


#     class Config:
#         arbitrary_types_allowed = True


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Initialize resources on startup and cleanup on shutdown"""
#     global chain, schema

#     try:
#         # Initialize database and schema
#         logger.info(f'Initializing database at {DB_PATH}')
#         # init_db(DB_PATH)
#         # insert_sample_data(DB_PATH)
#         schema = fetch_schema(DB_PATH)
#         logger.info('Database initialized successfully!')

#         logger.info('Building LLM chain')
#         chain = build_chain()
#         logger.info('LLM chain built successfully!')
    
#     except Exception as e:
#         logger.error(f'Failed to initialize application: {e}')
#         raise

#     yield

#     logger.info('Application shutdown')

# app = FastAPI(
#     title='NL2SQL API',
#     description='Natural Language to SQL query generation and execution service',
#     version='1.0.0',
#     lifespan=lifespan
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_methods=['GET', 'POST'],
#     allow_headers=['*'],
# )


# @app.exception_handler(Exception)
# async def global_execution_handler(request, exc):
#     logger.error(f'Unhandled exception: {exc}', exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={'error': 'Internal server error'}
#     )

# def get_chain():
#     """Dependency to get the LLM chain"""
#     if chain is None:
#         raise HTTPException(status_code=503, detail="Service not ready - LLM chain not initialized")
#     return chain

# def get_schema():
#     """Dependency to get the database schema"""
#     if schema is None:
#         raise HTTPException(status_code=503, detail="Service not ready - Database schema not loaded")
#     return schema

# @app.get('/api/health')
# def health_check():
#     """Health check endpoint"""
#     return {
#         'status':'healthy',
#         'database_initialized': schema is not None,
#         'llm_chain_initialized': chain is not None
#     }

# @app.get("/api/test")
# def test_data():
#     cols, rows = safe_select("SELECT * FROM customers", DB_PATH)
#     return {
#         'columns': cols,
#         'rows': rows
#     }


# @app.post('/api/nl2sql', response_model=SQLResponse)
# def nl2sql(
#     query: GenerateSQLQuery,
#     llm_chain=Depends(get_chain),
#     db_schema=Depends(get_schema)
# ):
#     """Convert natural language question to SQL and optionally executes it"""
#     try:
#         logger.info(f"Processing question: {query.question}")

#         # Generate SQL using LLM chain
#         response = llm_chain.invoke({'schema':db_schema, 'question':query.question})

#         sql = extract_sql(response)
#         if not sql:
#             raise HTTPException(status_code=400, detail='Could not generate valid SQL from question')
        
#         logger.info(f'Generated SQL: {sql}')
#         result = SQLResponse(sql=sql)

#         # Execute SQL if requested
#         if query.execute:
#             try:
#                 columns, rows = safe_select(sql, DB_PATH)
#                 if rows and len(rows) > 0:
#                     table = PrettyTable()
#                     table.field_names = columns
#                     for row in rows:
#                         table.add_row(row)
#                     formatted = str(table)
#                 else:
#                     formatted = "No data found"

#                 result.columns = columns
#                 result.rows = rows
#                 result.table = formatted
#                 logger.info(f"SQL executed successfully, returned {len(rows)} rows")
#             except Exception as e:
#                 error_msg = f"SQL execution error: {str(e)}"
#                 logger.error(error_msg)
#                 result.message = error_msg
#         return result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f'Error processing request: {e}', exc_info=True)
#         raise HTTPException(status_code=500, detail='Failed to process request')



# @app.get("/")
# def read_root():
#     return FileResponse('frontend/dist/index.html')

# # Mount static files (frontend)
# app.mount("/static", StaticFiles(directory='frontend/dist', html=True), name='static')

# if __name__ == '__main__':
#     uvicorn.run('main:app', host=HOST, port=PORT, log_level=LOG_LEVEL.lower(), reload=True)