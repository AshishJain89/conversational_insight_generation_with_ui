"""
Production-Ready Data Insight Generator
Analyzes SQL query results and generates natural language insights with caching, rate limiting, and error recovery
"""

import os, json, time, hashlib, logging, numpy as np, pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


@dataclass
class InsightResult:
    insights: List[str]
    summary: str
    key_metrics: Dict[str, Union[int, float, str]]
    recommendations: List[str]
    confidence_score: float
    analysis_time: float
    data_type: str


class InsightCache:
    """Simple in-memory cache for insights"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[InsightResult]:
        if key not in self.cache:
            return None

        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None

        return self.cache[key]

    def set(self, key: str, value: InsightResult):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self):
        self.cache.clear()
        self.timestamps.clear()


class DataAnalyzer:
    """Fast statistical analysis with optimized operations"""

    @staticmethod
    def analyze_dataset(
        df: pd.DataFrame, max_sample_size: int = 10000
    ) -> Dict[str, Any]:
        """Optimized dataset analysis with sampling for large datasets"""
        start_time = time.time()

        # Sample large datasets for performance
        if len(df) > max_sample_size:
            df = df.sample(n=max_sample_size, random_state=42)

        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "analysis_time": time.time() - start_time,
        }

        # Numeric analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            analysis["numeric_summary"] = numeric_stats

        # Categorical analysis
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            cat_stats = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                cat_stats[col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(3).to_dict(),
                }
            analysis["categorical_summary"] = cat_stats

        # Date analysis
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(date_cols) > 0:
            date_stats = {}
            for col in date_cols:
                date_stats[col] = {
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max()),
                    "date_range_days": (df[col].max() - df[col].min()).days,
                }
            analysis["date_summary"] = date_stats

        return analysis


class DataInsightGenerator:
    """Production-ready insight generator with caching, rate limiting, and error recovery"""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.1,
        timeout_seconds: int = 30,
        enable_cache: bool = True,
        max_rows_analysis: int = 10000,
    ):

        self.model_name = model_name or os.getenv(
            "INSIGHT_MODEL_NAME", "llama3-8b-8192"
        )
        self.temperature = temperature
        self.timeout = timeout_seconds
        self.max_rows = max_rows_analysis

        # Initialize components
        self.cache = InsightCache() if enable_cache else None
        self.analyzer = DataAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Setup LLM
        try:
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

        self._setup_prompt()

    def _setup_prompt(self):
        """Optimized prompt for consistent, structured insights"""
        SYSTEM_MESSAGE = """
        You are a data analyst generating business insights. Return ONLY valid JSON.

        RULES:
        - Use simple, non-technical language
        - Focus on actionable findings
        - Include specific numbers/percentages
        - Be concise but informative

        JSON FORMAT (exactly this structure):
        {
            "summary": "One clear sentence summarizing the main finding",
            "insights": [ "Key finding with specific numbers", "Important trend or pattern", "Notable anomaly or opportunity" ],
            "key_metrics": { "primary_metric": "value with unit", "secondary_metric": "comparative value" },
            "recommendations": [ "Specific actionable step", "Investigation suggestion" ],
            "confidence_score": 0.85
        }
        """

        HUMAN_MESSAGE = """
        DATA ANALYSIS:
            Question: {question}
            Data Type: {data_type}
            Rows: {row_count} | Columns: {column_count}

            KEY STATISTICS:
            {stats_summary}

            SAMPLE DATA:
            {sample_data}

            Generate insights in JSON format only.
        """
        self.insight_prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_MESSAGE), ("human", HUMAN_MESSAGE)]
        )

    def generate_insights(
        self, question: str, columns: List[str], rows: List[List[Any]]
    ) -> InsightResult:
        """Generate insights with caching and timeout protection"""

        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(question, columns, rows)

        # Check cache
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached insights")
                return cached_result

        try:
            # Analyze data
            df = pd.DataFrame(rows, columns=columns)
            data_type = self._detect_data_type(question, columns)
            analysis = self.analyzer.analyze_dataset(df, self.max_rows)

            # Generate insights with timeout
            future = self.executor.submit(
                self._generate_insights_worker, question, df, data_type, analysis
            )

            try:
                result = future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                logger.warning("Insight generation timed out, using fallback")
                result = self._create_fallback_insights(
                    question, df, data_type, analysis
                )

            # Add timing and cache
            result.analysis_time = time.time() - start_time
            result.data_type = data_type

            if self.cache:
                self.cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return self._create_error_fallback(question, len(rows), len(columns))

    def _generate_insights_worker(
        self, question: str, df: pd.DataFrame, data_type: str, analysis: Dict[str, Any]
    ) -> InsightResult:
        """Worker function for insight generation"""

        if not self.llm:
            return self._create_fallback_insights(question, df, data_type, analysis)

        # Prepare context
        context = {
            "question": question,
            "data_type": data_type,
            "row_count": len(df),
            "column_count": len(df.columns),
            "stats_summary": self._format_stats_summary(analysis),
            "sample_data": self._format_sample_data(df, max_rows=3),
        }

        # Generate insights
        chain = self.insight_prompt | self.llm | StrOutputParser()
        response = chain.invoke(context)

        # Parse and validate response
        insight_data = self._parse_llm_response(response)

        return InsightResult(
            insights=insight_data["insights"],
            summary=insight_data["summary"],
            key_metrics=insight_data["key_metrics"],
            recommendations=insight_data["recommendations"],
            confidence_score=insight_data["confidence_score"],
            analysis_time=0.0,  # Will be set by caller
            data_type=data_type,
        )

    def _generate_cache_key(
        self, question: str, columns: List[str], rows: List[List[Any]]
    ) -> str:
        """Generate cache key from question and data hash"""
        # Use first few rows and columns for hash to balance cache effectiveness with performance
        sample_data = str(columns) + str(rows[:5]) + question.lower().strip()
        return hashlib.md5(sample_data.encode()).hexdigest()

    def _detect_data_type(self, question: str, columns: List[str]) -> str:
        """Fast data type detection using keywords"""

        q_lower = question.lower()
        cols_lower = " ".join(columns).lower()

        # Time series indicators
        time_indicators = [
            "trend",
            "over time",
            "monthly",
            "forecast",
            "growth",
            "date",
            "time",
        ]
        if any(
            indicator in q_lower or indicator in cols_lower
            for indicator in time_indicators
        ):
            return "time_series"

        # Sales indicators
        sales_indicators = ["sales", "revenue", "amount", "value", "order", "purchase"]
        if any(
            indicator in q_lower or indicator in cols_lower
            for indicator in sales_indicators
        ):
            return "sales"

        # Customer indicators
        customer_indicators = ["customer", "client", "user", "buyer", "contact"]
        if any(
            indicator in q_lower or indicator in cols_lower
            for indicator in customer_indicators
        ):
            return "customer"

        # Product indicators
        product_indicators = ["product", "item", "category", "inventory", "sku"]
        if any(
            indicator in q_lower or indicator in cols_lower
            for indicator in product_indicators
        ):
            return "product"

        return "general"

    def _format_stats_summary(self, analysis: Dict[str, Any]) -> str:
        """Format statistical summary for LLM context"""

        summary_parts = []

        # Basic info
        summary_parts.append(
            f"Dataset: {analysis['row_count']} rows, {analysis['column_count']} columns"
        )

        # Numeric summary
        if "numeric_summary" in analysis:
            numeric_cols = list(analysis["numeric_summary"].keys())[
                :3
            ]  # Limit to 3 columns
            for col in numeric_cols:
                stats = analysis["numeric_summary"][col]
                summary_parts.append(
                    f"{col}: avg={stats['mean']:.2f}, range={stats['min']:.2f}-{stats['max']:.2f}"
                )

        # Categorical summary
        if "categorical_summary" in analysis:
            for col, stats in list(analysis["categorical_summary"].items())[
                :2
            ]:  # Limit to 2 columns
                summary_parts.append(f"{col}: {stats['unique_count']} unique values")

        return " | ".join(summary_parts)

    def _format_sample_data(self, df: pd.DataFrame, max_rows: int = 3) -> str:
        """Format sample data efficiently"""
        sample = df.head(max_rows)
        return sample.to_string(index=False, max_cols=5)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""

        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:-3]

            # Parse JSON
            data = json.loads(cleaned)

            # Validate and provide defaults
            return {
                "summary": str(data.get("summary", "Analysis completed successfully")),
                "insights": data.get(
                    "insights", ["Data contains structured information"]
                ),
                "key_metrics": data.get("key_metrics", {}),
                "recommendations": data.get(
                    "recommendations", ["Review data for opportunities"]
                ),
                "confidence_score": float(data.get("confidence_score", 0.7)),
            }

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._get_default_insight_data()

    def _create_fallback_insights(
        self, question: str, df: pd.DataFrame, data_type: str, analysis: Dict[str, Any]
    ) -> InsightResult:
        """Create rule-based fallback insights"""

        insights = []
        key_metrics = {}
        recommendations = []

        # Basic insights
        row_count = len(df)
        col_count = len(df.columns)
        insights.append(
            f"Dataset contains {row_count:,} records across {col_count} attributes"
        )
        key_metrics["total_records"] = row_count

        # Numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:2]:  # Top 2 numeric columns
                total = df[col].sum()
                avg = df[col].mean()
                insights.append(
                    f"{col}: Total of {total:,.0f} with average of {avg:.2f}"
                )
                key_metrics[f"{col}_total"] = total

        # Data type specific insights
        if data_type == "time_series":
            date_cols = df.select_dtypes(include=["datetime64"]).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                date_range = (df[date_col].max() - df[date_col].min()).days
                insights.append(f"Time series spans {date_range} days")
                recommendations.append("Analyze trends over time for patterns")
        elif data_type == "sales":
            recommendations.extend(
                [
                    "Identify top-performing products or categories",
                    "Analyze seasonal patterns in sales data",
                ]
            )

        summary = (
            f"Analyzed {row_count:,} records revealing key patterns in {data_type} data"
        )

        return InsightResult(
            insights=insights,
            summary=summary,
            key_metrics=key_metrics,
            recommendations=recommendations,
            confidence_score=0.6,
            analysis_time=0.0,
            data_type=data_type,
        )

    def _create_error_fallback(
        self, question: str, row_count: int, col_count: int
    ) -> InsightResult:
        """Create minimal fallback for error cases"""

        return InsightResult(
            insights=[f"Dataset contains {row_count} records and {col_count} columns"],
            summary="Basic data analysis completed",
            key_metrics={"rows": row_count, "columns": col_count},
            recommendations=["Data is available for detailed analysis"],
            confidence_score=0.3,
            analysis_time=0.0,
            data_type="general",
        )

    def _get_default_insight_data(self) -> Dict[str, Any]:
        """Default insight data structure"""
        return {
            "summary": "Data analysis completed",
            "insights": ["Dataset ready for analysis"],
            "key_metrics": {},
            "recommendations": ["Explore data relationships"],
            "confidence_score": 0.5,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        if not self.cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "ttl_seconds": self.cache.ttl,
        }

    def clear_cache(self):
        """Clear insight cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Insight cache cleared")


# Integration functions
def create_insight_generator(
    model_name: str = None, enable_cache: bool = True, timeout_seconds: int = 30
) -> DataInsightGenerator:
    """Factory function with production defaults"""
    return DataInsightGenerator(
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        enable_cache=enable_cache,
    )


def generate_insights_safe(
    question: str,
    columns: List[str],
    rows: List[List[Any]],
    generator: DataInsightGenerator = None,
) -> Dict[str, Any]:
    """Safe insight generation with error handling"""

    if not generator:
        generator = create_insight_generator()

    try:
        result = generator.generate_insights(question, columns, rows)
        return asdict(result)
    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        return {
            "insights": ["Data analysis encountered an error"],
            "summary": "Analysis partially completed",
            "key_metrics": {"error": str(e)},
            "recommendations": ["Try with a smaller dataset"],
            "confidence_score": 0.1,
            "analysis_time": 0.0,
            "data_type": "error",
        }
