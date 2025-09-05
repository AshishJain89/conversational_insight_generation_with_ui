#!/usr/bin/env python3
"""
Test script for the new validation and feedback loop functionality
"""

import sqlite3
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_validator import create_validator
from feedback_loop import create_feedback_loop


def test_validator():
    """Test the query validator with various SQL queries"""
    print("Testing Query Validator...")

    # Create validator
    validator = create_validator("./data/northwind.db")

    # Test cases
    test_queries = [
        # Valid query
        "SELECT CustomerID, CompanyName FROM Customers LIMIT 5",
        # Invalid syntax - missing FROM
        "SELECT CustomerID, CompanyName LIMIT 5",
        # Invalid table
        "SELECT * FROM NonExistentTable LIMIT 5",
        # Invalid column
        "SELECT CustomerID, NonExistentColumn FROM Customers LIMIT 5",
        # Complex but valid query
        """SELECT c.CustomerID, c.CompanyName, COUNT(o.OrderID) as OrderCount
           FROM Customers c
           JOIN Orders o ON c.CustomerID = o.CustomerID
           GROUP BY c.CustomerID, c.CompanyName
           ORDER BY OrderCount DESC
           LIMIT 10""",
        # Query with performance issues
        "SELECT * FROM Orders ORDER BY OrderDate",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")

        try:
            result = validator.validate_sql(query)
            summary = validator.get_validation_summary(result)

            print(f"Valid: {result.is_valid}")
            print(f"Result: {result.result.value}")
            if result.error_message:
                print(f"Error: {result.error_message}")
            if result.suggestions:
                print(f"Suggestions: {', '.join(result.suggestions)}")
            if result.execution_time:
                print(f"Execution time: {result.execution_time:.4f}s")
            if result.complexity_score is not None:
                print(f"Complexity score: {result.complexity_score}")

        except Exception as e:
            print(f"Error during validation: {e}")

    print("\n" + "=" * 50)


def test_feedback_loop():
    """Test the feedback loop with a complex question"""
    print("Testing Feedback Loop...")

    try:
        # Create feedback loop
        feedback_loop = create_feedback_loop("./data/northwind.db", max_retries=1)

        # Test with a complex question
        complex_question = "Show me the top 5 customers by total order value in 2022, including their contact info and order count"

        print(f"Question: {complex_question}")
        print("Processing with feedback loop...")

        result = feedback_loop.process_query_with_feedback(complex_question)

        print(f"Success: {result['success']}")
        print(f"Attempts: {result['attempts']}")
        print(f"Final SQL: {result['final_sql']}")

        if result["validation_results"]:
            print("\nValidation Results:")
            for i, val_result in enumerate(result["validation_results"]):
                print(f"  Attempt {i+1}: {val_result['validation']['result']}")
                if not val_result["validation"]["is_valid"]:
                    print(f"    Error: {val_result['validation']['error_message']}")

        # Get performance metrics
        metrics = feedback_loop.get_performance_metrics()
        if metrics:
            print(f"\nPerformance Metrics:")
            print(f"  Total queries: {metrics['total_queries']}")
            print(f"  Success rate: {metrics['success_rate_percent']}%")
            print(f"  Average attempts: {metrics['average_attempts_per_query']}")

    except Exception as e:
        print(f"Error during feedback loop test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)


def test_schema_loading():
    """Test that the schema is properly loaded"""
    print("Testing Schema Loading...")

    try:
        validator = create_validator("./data/northwind.db")

        if validator.schema_cache:
            print(f"Schema loaded successfully!")
            print(f"Tables: {list(validator.schema_cache.keys())}")
            print(
                f"Total columns: {sum(len(cols) for cols in validator.schema_cache.values())}"
            )

            # Show some sample columns
            for table, columns in list(validator.schema_cache.items())[:3]:
                print(f"  {table}: {columns[:5]}...")
        else:
            print("Failed to load schema")

    except Exception as e:
        print(f"Error loading schema: {e}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    print("Testing New Validation and Feedback Loop System")
    print("=" * 50)

    # Test schema loading first
    test_schema_loading()

    # Test validator
    test_validator()

    # Test feedback loop
    test_feedback_loop()

    print("Testing completed!")
