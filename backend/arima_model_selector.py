"""
Standalone ARIMA Model Selector
Improved model selection algorithm for time series forecasting
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ARIMAModelSelector:
    """
    Advanced ARIMA model selector with better parameter selection
    and model validation
    """
    
    def __init__(self, use_auto_arima: bool = True):
        self.use_auto_arima = use_auto_arima
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are available"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.ARIMA = ARIMA
        except ImportError:
            raise ImportError("statsmodels ARIMA not available")
        
        try:
            from pmdarima import auto_arima
            self.auto_arima = auto_arima
            self._has_pmdarima = True
        except ImportError:
            self._has_pmdarima = False
            logger.warning("pmdarima not available, using fallback models")
    
    def select_and_fit(self, ts: pd.Series, seasonal: bool = None, m: int = None) -> Tuple:
        """
        Select and fit the best ARIMA model for the time series
        
        Args:
            ts: Time series data
            seasonal: Whether to use seasonal models
            m: Seasonal period (e.g., 12 for monthly)
            
        Returns:
            Tuple of (fitted_model, order, seasonal_order)
        """
        if len(ts) < 6:
            raise ValueError("Need at least 6 data points for forecasting")
        
        # Auto-detect seasonal parameters
        seasonal, m = self._detect_seasonality(ts, seasonal, m)
        
        # Try auto_arima first if available
        if self.use_auto_arima and self._has_pmdarima:
            try:
                model, order, seasonal_order = self._try_auto_arima(ts, seasonal, m)
                if self._validate_model_quality(model, ts):
                    return model, order, seasonal_order
            except Exception as e:
                logger.warning(f"Auto ARIMA failed: {e}")
        
        # Fallback to manual model selection
        return self._manual_model_selection(ts, seasonal, m)
    
    def _detect_seasonality(self, ts: pd.Series, seasonal: bool, m: int) -> Tuple[bool, int]:
        """Detect if seasonal models should be used"""
        if seasonal is not None and m is not None:
            return seasonal, m
        
        # Auto-detect based on data length and frequency
        if len(ts) >= 24:  # 2+ years
            seasonal = True
            m = 12
        elif len(ts) >= 12:  # 1+ year
            seasonal = True
            m = 12
        else:
            seasonal = False
            m = 1
        
        return seasonal, m
    
    def _try_auto_arima(self, ts: pd.Series, seasonal: bool, m: int) -> Tuple:
        """Try to fit model using auto_arima"""
        # For 24 months, be more conservative
        if len(ts) >= 24:
            max_p, max_q, max_d = 3, 3, 1  # Reduced from 5,5,2
            max_P, max_Q, max_D = 1, 1, 1  # Reduced from 2,2,1
            # Force d=1, D=1 to be more conservative
            d = 1
            D = 1 if seasonal else 0
        else:
            max_p, max_q, max_d = 2, 2, 1
            max_P, max_Q, max_D = 0, 0, 0
            d = 1
            D = 0
        
        arima = self.auto_arima(
            ts,
            seasonal=seasonal,
            m=m if seasonal else 1,
            error_action='warn',
            suppress_warnings=True,
            max_p=max_p, max_q=max_q, max_d=max_d,
            max_P=max_P, max_Q=max_Q, max_D=max_D,
            stepwise=True,
            information_criterion='aic',
            seasonal_test='ch',
            test='adf',
            start_p=1, start_q=1,
            start_P=1, start_Q=1,
            d=d, D=D,  # Force specific differencing
            trace=False
        )
        
        order = arima.order
        seasonal_order = arima.seasonal_order if seasonal else (0,0,0,0)
        model = self.ARIMA(ts, order=order, seasonal_order=seasonal_order).fit()
        
        return model, order, seasonal_order
    
    def _manual_model_selection(self, ts: pd.Series, seasonal: bool, m: int) -> Tuple:
        """Manual model selection with fallback models"""
        
        # Try seasonal models first if appropriate
        if seasonal and m == 12:
            seasonal_models = [
                (1,1,1,1,1,1,12),  # ARIMA(1,1,1)(1,1,1,12)
                (0,1,1,0,1,1,12),  # ARIMA(0,1,1)(0,1,1,12) - Seasonal MA
                (1,1,0,1,1,0,12),  # ARIMA(1,1,0)(1,1,0,12) - Seasonal AR
                (2,1,2,1,1,1,12),  # ARIMA(2,1,2)(1,1,1,12) - Complex seasonal
            ]
            
            for p, d, q, P, D, Q, s in seasonal_models:
                try:
                    model = self.ARIMA(ts, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()
                    if self._is_model_stable(model):
                        return model, (p,d,q), (P,D,Q,s)
                except Exception:
                    continue
        
        # Non-seasonal fallback models
        non_seasonal_models = [
            (1,1,1),  # ARIMA(1,1,1) - Basic trend model
            (2,1,2),  # ARIMA(2,1,2) - More complex trend
            (0,1,1),  # ARIMA(0,1,1) - Simple moving average
            (1,1,0),  # ARIMA(1,1,0) - Simple autoregressive
            (1,0,1),  # ARIMA(1,0,1) - ARMA without differencing
        ]
        
        for p, d, q in non_seasonal_models:
            try:
                model = self.ARIMA(ts, order=(p,d,q)).fit()
                if self._is_model_stable(model):
                    return model, (p,d,q), (0,0,0,0)
            except Exception:
                continue
        
        # Last resort: Simple but useful model
        try:
            model = self.ARIMA(ts, order=(1,1,1)).fit()
            return model, (1,1,1), (0,0,0,0)
        except Exception:
            raise RuntimeError('Failed to fit any ARIMA model')
    
    def _is_model_stable(self, model) -> bool:
        """Check if ARIMA model is stable and reasonable"""
        try:
            # Check if model has roots and they're within unit circle
            if hasattr(model, 'arroots') and hasattr(model, 'maroots'):
                ar_roots = model.arroots()
                ma_roots = model.maroots()
                
                # All roots should be inside unit circle for stability
                if np.any(np.abs(ar_roots) >= 1) or np.any(np.abs(ma_roots) >= 1):
                    return False
            
            # Check if residuals are reasonable
            if hasattr(model, 'resid'):
                residuals = model.resid
                if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                    return False
                
                # Check if residuals are too large relative to data
                if np.std(residuals) > 10 * np.std(model.model.endog):
                    return False
            
            # Check if AIC is reasonable (not too high)
            if hasattr(model, 'aic') and model.aic > 1000:
                return False
                
            return True
        except Exception:
            # If we can't check stability, assume it's okay
            return True
    
    def _validate_model_quality(self, model, ts: pd.Series) -> bool:
        """Validate that the model produces reasonable forecasts"""
        try:
            # Generate a short forecast to test
            test_forecast = model.get_forecast(steps=3)
            mean_values = test_forecast.predicted_mean
            
            # Check for reasonable bounds
            historical_mean = ts.mean()
            historical_std = ts.std()
            
            # Forecast should be within 3 standard deviations of historical mean
            lower_bound = historical_mean - 3 * historical_std
            upper_bound = historical_mean + 3 * historical_std
            
            if np.any(mean_values < lower_bound) or np.any(mean_values > upper_bound):
                return False
            
            # Check for flat forecasts (all values the same)
            if np.std(mean_values) < historical_std * 0.01:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            return False
    
    def prepare_series(self, columns: List[str], rows: List[List], 
                       date_col: str, value_col: str, 
                       freq: Optional[str] = None, 
                       missing_method: str = 'interpolate') -> Tuple[pd.Series, str]:
        """Prepare time series data for forecasting"""
        df = pd.DataFrame(rows, columns=columns)
        
        if date_col not in df.columns or value_col not in df.columns:
            raise ValueError('date_col or value_col missing from provided columns')

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            raise ValueError('No rows with valid dates found')

        df = df.set_index(date_col).sort_index()

        # Better frequency detection
        if not freq:
            if len(df) >= 2:
                time_diff = df.index[1] - df.index[0]
                days_diff = time_diff.days
                
                if 25 <= days_diff <= 35:  # Monthly
                    freq = 'ME'  # Month End
                elif 350 <= days_diff <= 380:  # Yearly
                    freq = 'Y'
                elif 1 <= days_diff <= 7:  # Daily
                    freq = 'D'
                else:
                    freq = 'ME'  # Default to monthly
        
        # Force monthly frequency for monthly data
        if freq == 'M' or freq == 'ME' or (freq is None and len(df) <= 24):
            freq = 'ME'
            # Resample to monthly if needed
            df = df.resample('ME').sum()
        
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


def test_model_selector():
    """Test the model selector with sample data"""
    # Create sample monthly data
    dates = pd.date_range('2021-01-01', '2022-12-31', freq='M')
    np.random.seed(42)
    values = 3000000 + np.random.normal(0, 200000, len(dates)) + \
             500000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)  # Add seasonality
    
    ts = pd.Series(values, index=dates)
    
    print(f"Testing with {len(ts)} months of data")
    print(f"Data range: {ts.min():,.0f} to {ts.max():,.0f}")
    print(f"Mean: {ts.mean():,.0f}")
    print(f"Std: {ts.std():,.0f}")
    
    # Test the selector
    selector = ARIMAModelSelector()
    
    try:
        model, order, seasonal_order = selector.select_and_fit(ts)
        print(f"\n✅ Model fitted successfully!")
        print(f"Order: {order}")
        print(f"Seasonal Order: {seasonal_order}")
        print(f"AIC: {model.aic:.2f}")
        
        # Test forecast
        forecast = model.get_forecast(steps=6)
        print(f"\nForecast for next 6 months:")
        for i, (date, value) in enumerate(zip(forecast.predicted_mean.index, forecast.predicted_mean.values)):
            print(f"  {date.strftime('%Y-%m')}: {value:,.0f}")
            
    except Exception as e:
        print(f"❌ Model fitting failed: {e}")


if __name__ == "__main__":
    test_model_selector()
