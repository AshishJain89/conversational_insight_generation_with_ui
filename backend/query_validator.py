import sqlite3, re, json, logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    VALID = "valid"
    INVALID_SYNTAX = "invalid_syntax"
    INVALID_TABLE = "invalid_table"
    INVALID_COLUMN = "invalid_column"
    INVALID_JOIN = "invalid_join"
    EXECUTION_ERROR = "execution_error"
    PERFORMANCE_ISSUE = "performance_issue"

@dataclass
class ValidationFeedback:
    is_valid: bool
    result: ValidationResult
    error_message: Optional[str] = None
    suggestions: List[str] = None
    execution_time: Optional[float] = None
    row_count: Optional[int] = None
    complexity_score: Optional[int] = None

class QueryValidator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema_cache = None
        self._load_schema()
    
    def _load_schema(self):
        """Load and cache database schema for validation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get schema for each table
            schema = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                schema[table] = columns
            
            self.schema_cache = schema
            conn.close()
            logger.info(f"Schema loaded: {len(tables)} tables, {sum(len(cols) for cols in schema.values())} columns")
            
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            self.schema_cache = {}
    
    def validate_sql(self, sql: str) -> ValidationFeedback:
        """Comprehensive SQL validation with detailed feedback"""
        if not sql or not sql.strip():
            return ValidationFeedback(
                is_valid=False,
                result=ValidationResult.INVALID_SYNTAX,
                error_message="Empty SQL query"
            )
        
        # Basic syntax validation
        syntax_check = self._validate_syntax(sql)
        if not syntax_check.is_valid:
            return syntax_check
        
        # Schema validation
        schema_check = self._validate_schema(sql)
        if not schema_check.is_valid:
            return schema_check
        
        # Execution validation
        execution_check = self._validate_execution(sql)
        if not execution_check.is_valid:
            return execution_check
        
        # Performance analysis
        performance_check = self._analyze_performance(sql)
        
        return ValidationFeedback(
            is_valid=True,
            result=ValidationResult.VALID,
            execution_time=execution_check.execution_time,
            row_count=execution_check.row_count,
            complexity_score=performance_check.complexity_score,
            suggestions=performance_check.suggestions
        )
    
    def _validate_syntax(self, sql: str) -> ValidationFeedback:
        """Basic SQL syntax validation"""
        sql_upper = sql.upper().strip()
        
        # Check if it starts with SELECT
        if not sql_upper.startswith('SELECT'):
            return ValidationFeedback(
                is_valid=False,
                result=ValidationResult.INVALID_SYNTAX,
                error_message="Query must start with SELECT",
                suggestions=["Ensure your query begins with SELECT"]
            )
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return ValidationFeedback(
                is_valid=False,
                result=ValidationResult.INVALID_SYNTAX,
                error_message="Unbalanced parentheses",
                suggestions=["Check for matching opening and closing parentheses"]
            )
        
        # Check for basic SQL structure
        required_keywords = ['FROM']
        for keyword in required_keywords:
            if keyword not in sql_upper:
                return ValidationFeedback(
                    is_valid=False,
                    result=ValidationResult.INVALID_SYNTAX,
                    error_message=f"Missing required keyword: {keyword}",
                    suggestions=[f"Add {keyword} clause to your query"]
                )
        
        return ValidationFeedback(is_valid=True, result=ValidationResult.VALID)
    
    def _validate_schema(self, sql: str) -> ValidationFeedback:
        """Validate tables and columns against actual schema"""
        if not self.schema_cache:
            return ValidationFeedback(is_valid=True, result=ValidationResult.VALID)
        
        # Extract table names from FROM and JOIN clauses
        table_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*|"[^"]*")'
        tables = re.findall(table_pattern, sql, re.IGNORECASE)
        
        # Clean table names (remove quotes)
        tables = [table.strip('"') for table in tables]
        
        # Check if all tables exist
        for table in tables:
            if table not in self.schema_cache:
                return ValidationFeedback(
                    is_valid=False,
                    result=ValidationResult.INVALID_TABLE,
                    error_message=f"Table '{table}' does not exist",
                    suggestions=[f"Available tables: {', '.join(self.schema_cache.keys())}"]
                )
        
        # Extract column references
        column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
        column_refs = re.findall(column_pattern, sql)
        
        # Check if all referenced columns exist
        for table_alias, column in column_refs:
            # Find the actual table name from the alias
            actual_table = self._find_table_by_alias(sql, table_alias, tables)
            if actual_table and actual_table in self.schema_cache:
                if column not in self.schema_cache[actual_table]:
                    return ValidationFeedback(
                        is_valid=False,
                        result=ValidationResult.INVALID_COLUMN,
                        error_message=f"Column '{column}' does not exist in table '{actual_table}'",
                        suggestions=[f"Available columns in {actual_table}: {', '.join(self.schema_cache[actual_table])}"]
                    )
        
        return ValidationFeedback(is_valid=True, result=ValidationResult.VALID)
    
    def _find_table_by_alias(self, sql: str, alias: str, tables: List[str]) -> Optional[str]:
        """Find the actual table name from an alias"""
        # Simple alias resolution - look for "table AS alias" or "table alias" patterns
        for table in tables:
            alias_pattern = rf'\b{table}\s+(?:AS\s+)?{re.escape(alias)}\b'
            if re.search(alias_pattern, sql, re.IGNORECASE):
                return table
        return None
    
    def _validate_execution(self, sql: str) -> ValidationFeedback:
        """Validate SQL by attempting execution with EXPLAIN"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use EXPLAIN to check if the query can be parsed
            cursor.execute(f"EXPLAIN {sql}")
            explain_result = cursor.fetchall()
            
            # Check if EXPLAIN succeeded
            if not explain_result:
                return ValidationFeedback(
                    is_valid=False,
                    result=ValidationResult.INVALID_SYNTAX,
                    error_message="Query cannot be parsed by SQLite",
                    suggestions=["Check SQL syntax and ensure all referenced objects exist"]
                )
            
            # Try to execute with LIMIT 1 to check for runtime errors
            limited_sql = self._add_limit_clause(sql, 1)
            start_time = self._get_time()
            cursor.execute(limited_sql)
            rows = cursor.fetchall()
            execution_time = self._get_time() - start_time
            
            conn.close()
            
            return ValidationFeedback(
                is_valid=True,
                result=ValidationResult.VALID,
                execution_time=execution_time,
                row_count=len(rows)
            )
            
        except sqlite3.Error as e:
            return ValidationFeedback(
                is_valid=False,
                result=ValidationResult.EXECUTION_ERROR,
                error_message=f"SQL execution error: {str(e)}",
                suggestions=self._get_error_suggestions(str(e))
            )
        except Exception as e:
            return ValidationFeedback(
                is_valid=False,
                result=ValidationResult.EXECUTION_ERROR,
                error_message=f"Unexpected error: {str(e)}",
                suggestions=["Check the query structure and try again"]
            )
    
    def _add_limit_clause(self, sql: str, limit: int) -> str:
        """Add LIMIT clause if not present"""
        sql_upper = sql.upper()
        if 'LIMIT' not in sql_upper:
            return f"{sql.rstrip(';')} LIMIT {limit};"
        return sql
    
    def _get_time(self) -> float:
        """Get current time for performance measurement"""
        import time
        return time.time()
    
    def _analyze_performance(self, sql: str) -> ValidationFeedback:
        """Analyze query performance and provide optimization suggestions"""
        suggestions = []
        complexity_score = 0
        
        sql_upper = sql.upper()
        
        # Count JOINs
        join_count = sql_upper.count('JOIN')
        if join_count > 3:
            complexity_score += 2
            suggestions.append("Consider breaking down complex multi-table joins into smaller queries")
        
        # Check for subqueries
        if '(' in sql and 'SELECT' in sql:
            complexity_score += 1
            suggestions.append("Consider using CTEs (WITH clause) instead of subqueries for better readability")
        
        # Check for ORDER BY without LIMIT
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            suggestions.append("Add LIMIT clause when using ORDER BY to improve performance")
        
        # Check for GROUP BY
        if 'GROUP BY' in sql_upper:
            complexity_score += 1
            suggestions.append("Ensure GROUP BY columns are properly indexed for better performance")
        
        # Check for LIKE with wildcard at start
        if 'LIKE' in sql_upper and re.search(r"LIKE\s+['\"]%", sql):
            suggestions.append("Avoid leading wildcards in LIKE clauses as they prevent index usage")
        
        return ValidationFeedback(
            is_valid=True,
            result=ValidationResult.VALID,
            complexity_score=complexity_score,
            suggestions=suggestions
        )
    
    def _get_error_suggestions(self, error_msg: str) -> List[str]:
        """Provide specific suggestions based on error messages"""
        error_lower = error_msg.lower()
        suggestions = []
        
        if 'no such table' in error_lower:
            suggestions.append("Check table name spelling and ensure the table exists")
        elif 'no such column' in error_lower:
            suggestions.append("Verify column names and check for typos")
        elif 'syntax error' in error_lower:
            suggestions.append("Review SQL syntax, check for missing keywords or incorrect punctuation")
        elif 'ambiguous column' in error_lower:
            suggestions.append("Use table aliases to disambiguate column references")
        elif 'near' in error_lower:
            suggestions.append("Check the area around the indicated syntax error")
        
        return suggestions or ["Review the query structure and try again"]
    
    def generate_feedback_prompt(self, original_question: str, failed_sql: str, 
                                validation_result: ValidationFeedback) -> str:
        """Generate a feedback prompt for the LLM to improve the query"""
        feedback_template = f"""
The following SQL query failed validation:

Question: {original_question}
Generated SQL: {failed_sql}
Error: {validation_result.error_message}

Please generate a corrected SQL query that addresses the validation error.
Consider the following suggestions: {', '.join(validation_result.suggestions) if validation_result.suggestions else 'None'}

Return only the corrected SQL query in JSON format with the same structure as before.
"""
        return feedback_template
    
    def get_validation_summary(self, feedback: ValidationFeedback) -> Dict[str, Any]:
        """Get a summary of validation results for logging/monitoring"""
        return {
            "is_valid": feedback.is_valid,
            "result": feedback.result.value,
            "error_message": feedback.error_message,
            "execution_time": feedback.execution_time,
            "row_count": feedback.row_count,
            "complexity_score": feedback.complexity_score,
            "suggestions": feedback.suggestions
        }

# Utility function for easy import
def create_validator(db_path: str) -> QueryValidator:
    """Factory function to create a QueryValidator instance"""
    return QueryValidator(db_path)
