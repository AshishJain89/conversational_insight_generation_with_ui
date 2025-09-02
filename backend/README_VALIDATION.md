# Query Validation and Feedback Loop System

This document describes the new validation and feedback loop system implemented to improve SQL query generation for complex questions.

## Overview

The system consists of two main components:
1. **QueryValidator** - Comprehensive SQL validation with detailed feedback
2. **QueryFeedbackLoop** - Intelligent retry mechanism with LLM feedback

## Features

### 1. QueryValidator (`query_validator.py`)

#### Capabilities
- **Syntax Validation**: Checks basic SQL structure, balanced parentheses, required keywords
- **Schema Validation**: Verifies tables and columns exist in the database
- **Execution Validation**: Tests queries with EXPLAIN and limited execution
- **Performance Analysis**: Identifies potential performance issues and provides optimization suggestions

#### Validation Results
```python
@dataclass
class ValidationFeedback:
    is_valid: bool
    result: ValidationResult  # Enum: VALID, INVALID_SYNTAX, INVALID_TABLE, etc.
    error_message: Optional[str]
    suggestions: List[str]
    execution_time: Optional[float]
    row_count: Optional[int]
    complexity_score: Optional[int]
```

#### Usage
```python
from query_validator import create_validator

validator = create_validator('./data/northwind.db')
result = validator.validate_sql("SELECT * FROM Customers")
print(f"Valid: {result.is_valid}")
print(f"Suggestions: {result.suggestions}")
```

### 2. QueryFeedbackLoop (`feedback_loop.py`)

#### Capabilities
- **Automatic Retry**: Attempts to fix failed queries up to configurable retry limit
- **LLM Feedback Integration**: Uses validation results to generate improved prompts
- **Performance Tracking**: Monitors success rates and retry efficiency
- **History Management**: Tracks query processing history for analysis

#### Usage
```python
from feedback_loop import create_feedback_loop

feedback_loop = create_feedback_loop('./data/northwind.db', max_retries=2)
result = feedback_loop.process_query_with_feedback("Show me top customers by sales")

print(f"Success: {result['success']}")
print(f"Attempts: {result['attempts']}")
print(f"Final SQL: {result['final_sql']}")
```

## API Endpoints

### New Endpoints

#### 1. `/api/validate-sql`
- **Method**: GET
- **Parameters**: `sql` (query string)
- **Purpose**: Validate SQL query and return detailed feedback
- **Response**: Validation results with suggestions

#### 2. `/api/feedback-metrics`
- **Method**: GET
- **Purpose**: Get performance metrics from feedback loop
- **Response**: Success rates, average attempts, validation failure analysis

#### 3. `/api/process-with-feedback`
- **Method**: POST
- **Purpose**: Process natural language question with feedback loop
- **Response**: Final SQL, validation results, execution results

### Enhanced Endpoint

#### `/api/nl2sql` (Enhanced)
- Now includes validation feedback
- Provides detailed error messages and suggestions
- Logs validation failures for monitoring

## Configuration

### Model Upgrade
The system now uses `llama-3.1-70b-instant` (70B parameters) instead of the previous 8B model for better complex query handling.

### Environment Variables
```bash
MODEL_NAME=llama-3.1-70b-instant  # Best free model available in Groq
```

### Retry Configuration
```python
feedback_loop = create_feedback_loop(db_path, max_retries=2)  # Configurable retry limit
```

## Integration

### Importing
```python
# Import the modules
from .query_validator import create_validator
from .feedback_loop import create_feedback_loop

# Use in your application
validator = create_validator(db_path)
feedback_loop = create_feedback_loop(db_path)
```

### Standalone Usage
Both modules can be used independently:
- Use `QueryValidator` for one-time validation
- Use `QueryFeedbackLoop` for complex query processing with retries
- Combine both for comprehensive query handling

## Testing

Run the test script to verify functionality:
```bash
cd backend
python test_validation.py
```

The test script validates:
- Schema loading
- Various SQL validation scenarios
- Feedback loop processing
- Performance metrics

## Benefits

### For Complex Queries
1. **Better Model**: 70B parameter model handles complex logic better
2. **Validation**: Catches errors before execution
3. **Feedback Loop**: Automatically retries and improves failed queries
4. **Performance Analysis**: Identifies optimization opportunities

### For Developers
1. **Detailed Feedback**: Specific error messages and suggestions
2. **Monitoring**: Track success rates and failure patterns
3. **Extensible**: Easy to add new validation rules
4. **Standalone**: Can be imported and used in other projects

## Error Handling

### Validation Errors
- **INVALID_SYNTAX**: Basic SQL structure issues
- **INVALID_TABLE**: Non-existent table references
- **INVALID_COLUMN**: Non-existent column references
- **INVALID_JOIN**: Join syntax or reference issues
- **EXECUTION_ERROR**: Runtime execution failures
- **PERFORMANCE_ISSUE**: Performance optimization suggestions

### Feedback Loop Errors
- **Max Retries Exceeded**: When all retry attempts fail
- **LLM Generation Failure**: When the model fails to generate improved SQL
- **Schema Loading Issues**: Database connection or schema parsing problems

## Performance Considerations

### Complexity Scoring
- **Score 0**: Simple queries (single table, basic SELECT)
- **Score 1**: Moderate complexity (GROUP BY, subqueries)
- **Score 2+**: High complexity (multiple JOINs, complex aggregations)

### Optimization Suggestions
- Use LIMIT with ORDER BY
- Avoid leading wildcards in LIKE clauses
- Consider CTEs instead of subqueries
- Break down complex multi-table joins

## Future Enhancements

1. **Learning from Failures**: Store failed queries for model training
2. **Query Templates**: Pre-built templates for common complex queries
3. **Performance Benchmarking**: Compare query execution times
4. **Schema Evolution**: Handle schema changes automatically
5. **Multi-Database Support**: Extend beyond SQLite

## Troubleshooting

### Common Issues

1. **Schema Loading Failures**
   - Check database path and permissions
   - Verify database file exists and is valid SQLite

2. **LLM Generation Failures**
   - Check API key and model availability
   - Verify network connectivity to Groq API

3. **Validation Errors**
   - Review error messages and suggestions
   - Check table and column names in schema

4. **Performance Issues**
   - Monitor complexity scores
   - Review optimization suggestions
   - Consider query simplification

### Debug Mode
Enable detailed logging by setting log level to DEBUG in your environment:
```bash
LOG_LEVEL=DEBUG
```

## Support

For issues or questions about the validation system:
1. Check the logs for detailed error messages
2. Run the test script to verify functionality
3. Review the validation results and suggestions
4. Check the performance metrics for patterns
