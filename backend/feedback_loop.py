import logging
from typing import Optional, Dict, Any, Tuple
from .query_validator import QueryValidator, ValidationFeedback, ValidationResult
from .llm_sql import build_chain

logger = logging.getLogger(__name__)


class QueryFeedbackLoop:
    def __init__(self, db_path: str, max_retries: int = 2):
        self.validator = QueryValidator(db_path)
        self.llm_chain = build_chain()
        self.max_retries = max_retries
        self.retry_history = []

    def process_query_with_feedback(
        self, question: str, initial_sql: str = None
    ) -> Dict[str, Any]:
        """
        Process a natural language question with feedback loop for failed queries

        Returns:
            Dict containing the final SQL, validation results, and retry information
        """
        if initial_sql:
            # Start with provided SQL
            current_sql = initial_sql
            attempt_number = 0
        else:
            # Generate initial SQL
            attempt_number = 0
            current_sql = self._generate_initial_sql(question)

        validation_results = []
        final_sql = None
        success = False

        for attempt in range(self.max_retries + 1):
            attempt_number = attempt

            # Validate current SQL
            validation = self.validator.validate_sql(current_sql)
            validation_results.append(
                {
                    "attempt": attempt_number,
                    "sql": current_sql,
                    "validation": self.validator.get_validation_summary(validation),
                }
            )

            if validation.is_valid:
                final_sql = current_sql
                success = True
                logger.info(
                    f"Query validated successfully on attempt {attempt_number + 1}"
                )
                break

            if attempt < self.max_retries:
                # Generate improved SQL based on validation feedback
                improved_sql = self._generate_improved_sql(
                    question, current_sql, validation
                )
                if improved_sql and improved_sql != current_sql:
                    current_sql = improved_sql
                    logger.info(
                        f"Generated improved SQL for attempt {attempt_number + 2}"
                    )
                else:
                    logger.warning(
                        f"Failed to generate improved SQL for attempt {attempt_number + 2}"
                    )
                    break
            else:
                logger.warning(
                    f"Max retries reached ({self.max_retries}) without success"
                )

        # Record retry history
        self.retry_history.append(
            {
                "question": question,
                "attempts": attempt_number + 1,
                "success": success,
                "validation_results": validation_results,
            }
        )

        return {
            "success": success,
            "final_sql": final_sql,
            "attempts": attempt_number + 1,
            "validation_results": validation_results,
            "retry_history": self.retry_history[-10:],  # Keep last 10 queries
        }

    def _generate_initial_sql(self, question: str) -> str:
        """Generate initial SQL from natural language question"""
        try:
            # Get schema from validator
            schema = self._get_schema_text()

            response = self.llm_chain.invoke({"schema": schema, "question": question})

            # Extract SQL from response
            sql = self._extract_sql_from_response(response)
            return sql

        except Exception as e:
            logger.error(f"Failed to generate initial SQL: {e}")
            return ""

    def _generate_improved_sql(
        self, question: str, failed_sql: str, validation: ValidationFeedback
    ) -> Optional[str]:
        """Generate improved SQL based on validation feedback"""
        try:
            # Create feedback prompt
            feedback_prompt = self.validator.generate_feedback_prompt(
                question, failed_sql, validation
            )

            # Get schema for context
            schema = self._get_schema_text()

            # Create enhanced prompt with feedback
            enhanced_prompt = f"""
{schema}

{feedback_prompt}

Please provide a corrected SQL query that addresses the validation issues.
Return only the corrected SQL query in JSON format with keys "sql", "forecast", and "error".
"""

            # Invoke LLM with enhanced prompt
            response = self.llm_chain.invoke(
                {"schema": schema, "question": enhanced_prompt}
            )

            # Extract SQL from response
            improved_sql = self._extract_sql_from_response(response)
            return improved_sql

        except Exception as e:
            logger.error(f"Failed to generate improved SQL: {e}")
            return None

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL from LLM response"""
        try:
            # Try to parse as JSON first
            import json

            if isinstance(response, str) and response.strip().startswith("{"):
                data = json.loads(response)
                if isinstance(data, dict) and "sql" in data and data["sql"]:
                    return data["sql"].strip()
        except Exception:
            pass

        # Fallback to regex extraction
        import re

        sql_match = re.search(r"SELECT[\s\S]+?;?$", response, re.IGNORECASE)
        if sql_match:
            return sql_match.group(0).strip()

        return ""

    def _get_schema_text(self) -> str:
        """Get schema text for LLM context"""
        try:
            # Import the schema fetching function from the main module
            from .db import fetch_schema

            schema = fetch_schema(self.validator.db_path)
            return schema
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return ""

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from retry history"""
        if not self.retry_history:
            return {}

        total_queries = len(self.retry_history)
        successful_queries = sum(1 for q in self.retry_history if q["success"])
        total_attempts = sum(q["attempts"] for q in self.retry_history)

        # Calculate average attempts per query
        avg_attempts = total_attempts / total_queries if total_queries > 0 else 0

        # Calculate success rate
        success_rate = (
            (successful_queries / total_queries * 100) if total_queries > 0 else 0
        )

        # Analyze validation failures
        validation_failures = {}
        for query in self.retry_history:
            for result in query["validation_results"]:
                if not result["validation"]["is_valid"]:
                    result_type = result["validation"]["result"]
                    validation_failures[result_type] = (
                        validation_failures.get(result_type, 0) + 1
                    )

        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate_percent": round(success_rate, 2),
            "total_attempts": total_attempts,
            "average_attempts_per_query": round(avg_attempts, 2),
            "validation_failures": validation_failures,
            "retry_efficiency": (
                round(successful_queries / total_attempts * 100, 2)
                if total_attempts > 0
                else 0
            ),
        }

    def reset_history(self):
        """Reset retry history"""
        self.retry_history = []
        logger.info("Query feedback loop history reset")

    def get_retry_suggestions(self, validation_result: ValidationFeedback) -> list:
        """Get specific suggestions for improving failed queries"""
        suggestions = []

        if validation_result.result == ValidationResult.INVALID_SYNTAX:
            suggestions.extend(
                [
                    "Review SQL syntax and ensure proper keyword placement",
                    "Check for balanced parentheses and quotes",
                    "Verify all required clauses (SELECT, FROM) are present",
                ]
            )

        elif validation_result.result == ValidationResult.INVALID_TABLE:
            suggestions.extend(
                [
                    "Verify table names match exactly with the database schema",
                    "Check for typos in table names",
                    "Ensure table names are properly quoted if they contain spaces",
                ]
            )

        elif validation_result.result == ValidationResult.INVALID_COLUMN:
            suggestions.extend(
                [
                    "Verify column names exist in the referenced tables",
                    "Use table aliases to disambiguate column references",
                    "Check for typos in column names",
                ]
            )

        elif validation_result.result == ValidationResult.INVALID_JOIN:
            suggestions.extend(
                [
                    "Ensure proper JOIN syntax with ON clauses",
                    "Verify join conditions reference existing columns",
                    "Check that all joined tables are properly referenced",
                ]
            )

        elif validation_result.result == ValidationResult.EXECUTION_ERROR:
            suggestions.extend(
                [
                    "Test the query with EXPLAIN to identify parsing issues",
                    "Check for runtime errors like division by zero",
                    "Verify data types in WHERE clauses match column definitions",
                ]
            )

        # Add performance suggestions if available
        if validation_result.suggestions:
            suggestions.extend(validation_result.suggestions)

        return suggestions


# Utility function for easy import
def create_feedback_loop(db_path: str, max_retries: int = 2) -> QueryFeedbackLoop:
    """Factory function to create a QueryFeedbackLoop instance"""
    return QueryFeedbackLoop(db_path, max_retries)
