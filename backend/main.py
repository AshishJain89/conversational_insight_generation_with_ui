from prettytable import PrettyTable
import os, logging, uvicorn, sqlite3, json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, List, Any, Dict

from .db import fetch_schema, safe_select
from .llm_sql import build_chain, extract_sql, detect_forecast_intent
from .query_validator import create_validator
from .feedback_loop import create_feedback_loop
from .arima_model_selector import ARIMAModelSelector

# forecasting imports
import pandas as pd, numpy as np, io, base64, traceback, matplotlib, matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

try:
    from pmdarima import auto_arima
    _HAS_PMDARIMA = True
except Exception:
    _HAS_PMDARIMA = False

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

matplotlib.use('Agg')

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
validator = None
feedback_loop = None

class GenerateSQLQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description='Natural language question')
    execute: bool = Field(default=True, description='Whether to execute the generated SQL')

class SQLResponse(BaseModel):
    sql: Optional[str] = None
    forecast: Optional[bool] = None
    table: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    message: Optional[str] = None
    type: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, schema, validator, feedback_loop
    try:
        logger.info(f'Initializing database at {DB_PATH}')
        schema = fetch_schema(DB_PATH)
        logger.info('Database initialized successfully!')

        logger.info('Building LLM chain')
        chain = build_chain()
        logger.info('LLM chain built successfully!')
        
        logger.info('Initializing query validator and feedback loop')
        validator = create_validator(DB_PATH)
        feedback_loop = create_feedback_loop(DB_PATH, max_retries=2)
        logger.info('Query validator and feedback loop initialized successfully!')
    except Exception as e:
        logger.error(f'Failed to initialize application: {e}')
        raise
    yield
    logger.info('Application shutdown')

app = FastAPI(
    title='Conversational Insight Generator',
    description='Natural Language to SQL query generation and execution service',
    version='1.0.0',
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['GET','POST'],
    allow_headers=['*']
)

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

def get_validator():
    if validator is None:
        raise HTTPException(status_code=503, detail="Service not ready - Query validator not initialized")
    return validator

def get_feedback_loop():
    if feedback_loop is None:
        raise HTTPException(status_code=503, detail="Service not ready - Feedback loop not initialized")
    return feedback_loop

@app.get('/api/health')
def health_check():
    return {
        'status':'healthy',
        'database_initialized': schema is not None,
        'llm_chain_initialized': chain is not None,
        'validator_initialized': validator is not None,
        'feedback_loop_initialized': feedback_loop is not None
    }

@app.get('/api/test')
def test_data():
    cols, rows = safe_select('SELECT * FROM customers LIMIT 5', DB_PATH)
    return {'columns': cols, 'rows': rows}

@app.get('/api/validate-sql')
def validate_sql_endpoint(sql: str, db_validator=Depends(get_validator)):
    """Validate SQL query and return detailed feedback"""
    try:
        validation_result = db_validator.validate_sql(sql)
        return {
            'sql': sql,
            'validation': db_validator.get_validation_summary(validation_result)
        }
    except Exception as e:
        logger.error(f'SQL validation error: {e}')
        raise HTTPException(status_code=500, detail='Failed to validate SQL')

@app.get('/api/feedback-metrics')
def get_feedback_metrics(db_feedback_loop=Depends(get_feedback_loop)):
    """Get performance metrics from the feedback loop"""
    try:
        return db_feedback_loop.get_performance_metrics()
    except Exception as e:
        logger.error(f'Failed to get feedback metrics: {e}')
        raise HTTPException(status_code=500, detail='Failed to get feedback metrics')

@app.post('/api/process-with-feedback')
def process_with_feedback(
    query: GenerateSQLQuery,
    db_feedback_loop=Depends(get_feedback_loop)
):
    """Process natural language question with feedback loop for failed queries"""
    try:
        logger.info(f"Processing question with feedback loop: {query.question}")
        
        result = db_feedback_loop.process_query_with_feedback(query.question)
        
        # If successful and execution is requested, execute the final SQL
        if result['success'] and query.execute and result['final_sql']:
            try:
                columns, rows = safe_select(result['final_sql'], DB_PATH)
                result['execution_result'] = {
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows)
                }
                logger.info(f"Feedback loop SQL executed successfully, returned {len(rows)} rows")
            except Exception as e:
                logger.exception('SQL execution error in feedback loop')
                result['execution_error'] = str(e)
        
        return result
        
    except Exception as e:
        logger.error(f'Error processing with feedback: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to process with feedback')

# Updated nl2sql endpoint in main.py
@app.post('/api/nl2sql')
def nl2sql(
    query: GenerateSQLQuery,
    llm_chain=Depends(get_chain),
    db_schema=Depends(get_schema),
    db_validator=Depends(get_validator)
):
    try:
        logger.info(f"Processing question: {query.question}")

        # Generate SQL via LLM
        response = llm_chain.invoke({'schema': db_schema, 'question': query.question})

        # Parse response to extract SQL and forecast intent
        response_dict = {}
        sql = None
        forecast = False
        
        try:
            if isinstance(response, str) and response.strip().startswith('{'):
                response_dict = json.loads(response)
                sql = response_dict.get('sql')
                forecast = response_dict.get('forecast', False)
            
            if not sql:
                sql = extract_sql(response)
                
            # Also check for forecast intent in the original question
            if not forecast:
                forecast = detect_forecast_intent(query.question)
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            sql = extract_sql(response) if response else None
            forecast = detect_forecast_intent(query.question)

        # Initialize result object
        result = SQLResponse(sql=sql, forecast=forecast, type="table")

        # Always try to execute SQL if we have it and execution is requested
        if query.execute and sql:
            # Validate the SQL first
            validation_feedback = db_validator.validate_sql(sql)
            if not validation_feedback.is_valid:
                result.message = f"Generated SQL failed validation: {validation_feedback.error_message}"
                if validation_feedback.suggestions:
                    result.message += f". Suggestions: {', '.join(validation_feedback.suggestions)}"
                result.type = "text"
                return result.model_dump()
            
            try:
                # Execute the SQL query
                columns, rows = safe_select(sql, DB_PATH)
                result.columns = columns
                result.rows = rows
                
                # Set appropriate type and message based on forecast intent
                if forecast:
                    result.type = "forecast"
                    result.message = f"Retrieved {len(rows)} rows for forecasting analysis. Click 'Generate Forecast' to run ARIMA model."
                else:
                    # Check if it's time-series data but not forecasting
                    has_date_column = any('date' in col.lower() or 'time' in col.lower() or 'month' in col.lower() or 'year' in col.lower() for col in columns)
                    if has_date_column:
                        result.type = "forecast"  # Use forecast type for historical time-series
                        result.message = f"Retrieved {len(rows)} rows of historical time-series data. You can analyze trends or generate forecasts."
                    else:
                        # Regular query - create pretty table
                        table = PrettyTable() if rows else None
                        if table and len(rows) > 0:
                            table.field_names = columns
                            # Limit table display to first 50 rows for performance
                            display_rows = rows[:50]
                            for row in display_rows:
                                table.add_row(row)
                            result.table = str(table)
                            if len(rows) > 50:
                                result.message = f"Showing first 50 of {len(rows)} rows"
                        else:
                            result.table = 'No data found'
                            result.message = 'Query executed successfully but returned no results'
                
                logger.info(f"SQL executed successfully, returned {len(rows)} rows")
                
            except Exception as e:
                logger.exception('SQL execution error')
                result.message = f"SQL execution error: {e}"
                result.type = "text"
                result.sql = sql  # Keep the SQL even if execution failed
        
        elif not sql:
            result.message = 'No valid SQL generated. Please rephrase your question.'
            result.type = 'text'
        
        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error processing request: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Failed to process request')

# ---- Forecast endpoint ----
MAX_HORIZON = 365
MAX_ROWS = 20000
DEFAULT_TIMEOUT = 30
EXECUTOR = ThreadPoolExecutor(max_workers=2)

class ForecastRequest(BaseModel):
    sql: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = Field(default=None, description="Row-major data matching columns")
    horizon: int = 12
    freq: Optional[str] = None
    seasonal: bool = False
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    missing_method: Optional[str] = Field('interpolate', description='interpolate|ffill|bfill|drop')

class ForecastPoint(BaseModel):
    date: str
    mean: float
    lower: float
    upper: float

class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]
    model: Dict[str, Any]
    diagnostics: Dict[str, Any]
    plot_png_base64: Optional[str]
    error: Optional[str] = None


def _detect_columns(columns: List[str], df: pd.DataFrame, date_col: Optional[str], value_col: Optional[str]):
    if date_col and date_col not in columns:
        raise ValueError(f"date_col '{date_col}' not present in columns")
    if value_col and value_col not in columns:
        raise ValueError(f"value_col '{value_col}' not present in columns")

    if not date_col:
        for c in columns:
            if pd.to_datetime(df[c], errors='coerce').notna().any():
                date_col = c
                break
    if not value_col:
        for c in columns:
            if c == date_col:
                continue
            if pd.to_numeric(df[c], errors='coerce').notna().any():
                value_col = c
                break
    if not date_col or not value_col:
        raise ValueError('Could not autodetect date_col and/or value_col. Please provide them.')
    return date_col, value_col


def _prepare_series(columns: List[str], rows: List[List[Any]], date_col: str, value_col: str, freq: Optional[str], missing_method: str):
    df = pd.DataFrame(rows, columns=columns)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError('date_col or value_col missing from provided columns')

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    if df.empty:
        raise ValueError('No rows with valid dates found')

    df = df.set_index(date_col).sort_index()

    # Better frequency detection for monthly data
    if not freq:
        # Check if data looks monthly
        if len(df) >= 2:
            time_diff = df.index[1] - df.index[0]
            if time_diff.days >= 25 and time_diff.days <= 35:  # Roughly monthly
                freq = 'ME'  # Changed from 'M' to 'ME'
            elif time_diff.days >= 350 and time_diff.days <= 380:  # Roughly yearly
                freq = 'Y'
            else:
                freq = 'D'  # Default to daily
    
    # Force monthly frequency for monthly data
    if freq == 'M' or freq == 'ME' or (freq is None and len(df) <= 24):
        freq = 'ME'  # Changed from 'M' to 'ME'
        # Resample to monthly if needed
        df = df.resample('ME').sum()  # Changed from 'M' to 'ME'
    
    series = df[value_col]
    
    # Handle missing values
    na_frac = series.isna().mean()
    if na_frac > 0:
        if missing_method == 'interpolate':
            series = series.interpolate()
        elif missing_method == 'ffill':
            series = series.fillna(method='ffill')
        elif missing_method == 'bfill':
            series = series.fillna(method='bfill')
        elif missing_method == 'drop':
            series = series.dropna()
        else:
            series = series.interpolate()

    if series.empty:
        raise ValueError('Time series empty after preprocessing')

    return series, freq


def _select_and_fit(ts: pd.Series, seasonal: bool, m: int):
    selector = ARIMAModelSelector()
    return selector.select_and_fit(ts, seasonal, m)


def _forecast_and_plot(model, ts: pd.Series, horizon: int, freq: str):
    pred = model.get_forecast(steps=horizon)
    mean = pred.predicted_mean
    ci = pred.conf_int()

    # Validate forecast values before plotting
    if np.any(np.isnan(mean)) or np.any(np.isinf(mean)):
        raise ValueError("Forecast contains NaN or infinite values - model is unstable")
    
    # Check for reasonable bounds (not too extreme)
    max_historical = ts.max()
    min_historical = ts.min()
    reasonable_range = (min_historical * 0.1, max_historical * 10)
    
    if np.any(mean < reasonable_range[0]) or np.any(mean > reasonable_range[1]):
        raise ValueError("Forecast values are outside reasonable bounds - model may be overfitting")

    last = ts.index[-1]
    try:
        start = last + pd.tseries.frequencies.to_offset(freq)
    except Exception:
        start = last + pd.tseries.frequencies.to_offset('D')
        freq = 'D'
    future_idx = pd.date_range(start, periods=horizon, freq=freq)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts.index, ts.values, label='history')
    ax.plot(future_idx, mean.values, label='forecast')
    ax.fill_between(future_idx, ci.iloc[:,0], ci.iloc[:,1], color='gray', alpha=0.25)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()

    forecast_list = []
    for d, m, (low, high) in zip(future_idx, mean.values, ci.values):
        # Ensure all values are finite and within reasonable bounds
        if np.isfinite(m) and np.isfinite(low) and np.isfinite(high):
            forecast_list.append({
                'date': pd.Timestamp(d).strftime('%Y-%m-%d'),
                'mean': float(m),
                'lower': float(low),
                'upper': float(high)
            })

    if not forecast_list:
        raise ValueError("No valid forecast values generated")

    return forecast_list, img_b64


@app.post('/api/forecast')
def api_forecast(req: ForecastRequest):
    if req.horizon > MAX_HORIZON:
        raise HTTPException(status_code=400, detail=f'horizon too large, max {MAX_HORIZON}')

    def job():
        try:
            # Determine data source: SQL or provided rows/columns
            if req.sql:
                columns, rows = safe_select(req.sql, DB_PATH)
            else:
                if req.columns is None or req.rows is None:
                    raise ValueError('Either sql must be provided, or columns and rows must be supplied')
                columns, rows = req.columns, req.rows

            # More reasonable data requirements
            if len(rows) < 6:  # Need at least 6 months for basic forecasting
                return {
                    'forecast': [], 
                    'model': {}, 
                    'diagnostics': {}, 
                    'plot_png_base64': None, 
                    'error': 'Insufficient data for reliable forecasting. Need at least 6 data points for monthly data.'
                }

            if len(rows) > MAX_ROWS:
                raise ValueError(f'too many rows, max {MAX_ROWS}')

            df = pd.DataFrame(rows, columns=columns)
            date_col, value_col = _detect_columns(columns, df, req.date_col, req.value_col)
            ts, freq = _prepare_series(columns, rows, date_col, value_col, req.freq, req.missing_method)
            
            # Adjust seasonal parameters based on data length
            if len(rows) >= 24:
                seasonal_m = 12  # Full seasonal cycle
                req.seasonal = True
            elif len(rows) >= 12:
                seasonal_m = 12  # Try seasonal, but may fall back
                req.seasonal = True
            else:
                seasonal_m = 1   # No seasonal component for short series
                req.seasonal = False
            
            model, order, seasonal_order = _select_and_fit(ts, req.seasonal, seasonal_m)

            forecast_list, img_b64 = _forecast_and_plot(model, ts, req.horizon, freq)

            diagnostics = {
                'aic': float(getattr(model, 'aic', np.nan)),
                'bic': float(getattr(model, 'bic', np.nan)),
                'residuals_mean': float(np.mean(model.resid)) if hasattr(model, 'resid') else None,
                'rmse': float(np.sqrt(np.mean(np.square(model.resid)))) if hasattr(model, 'resid') else None
            }

            model_info = {'order': order, 'seasonal_order': seasonal_order}

            return {'forecast': forecast_list, 'model': model_info, 'diagnostics': diagnostics, 'plot_png_base64': img_b64, 'error': None}
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception('Forecast job error')
            return {'forecast': [], 'model': {}, 'diagnostics': {}, 'plot_png_base64': None, 'error': str(e)}

    fut = EXECUTOR.submit(job)
    try:
        res = fut.result(timeout=DEFAULT_TIMEOUT)
    except FuturesTimeoutError:
        fut.cancel()
        raise HTTPException(status_code=504, detail='Forecasting timed out')

    if res.get('error'):
        raise HTTPException(status_code=400, detail=res.get('error'))

    return res

@app.get('/')
def read_root():
    return FileResponse('frontend/dist/index.html')

app.mount('/static', StaticFiles(directory='frontend/dist', html=True), name='static')

if __name__ == '__main__':
    uvicorn.run('main:app', host=HOST, port=PORT, log_level=LOG_LEVEL.lower(), reload=True)