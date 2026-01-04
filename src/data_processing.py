"""
Data Processing Module for FitPulse Health Anomaly Detection
Complementary functions to support the main preprocessing pipeline
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import pytz
from typing import Union, Optional, Dict, List


def load_fitness_data(file_path: str, file_type: str = 'auto') -> pd.DataFrame:
    """
    Universal data loader supporting multiple formats
    Complements the DataLoader class in app.py
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    file_type : str
        'csv', 'json', 'excel', or 'auto'
    
    Returns:
    --------
    pandas.DataFrame
        Loaded fitness data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect file type from extension
    if file_type == 'auto':
        file_type = file_path.suffix.lower().replace('.', '')
    
    try:
        if file_type in ['csv', 'txt']:
            df = pd.read_csv(file_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=1).columns else False)
        elif file_type == 'json':
            df = load_json_fitness_data(file_path)
        elif file_type in ['xlsx', 'xls', 'excel']:
            df = pd.read_excel(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if df.empty:
            raise ValueError("Loaded data is empty")
        
        return df
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def load_json_fitness_data(file_path: str) -> pd.DataFrame:
    """
    Load fitness data from nested JSON format
    Supports various JSON structures commonly used in fitness apps
    
    Parameters:
    -----------
    file_path : str
        Path to JSON file
    
    Returns:
    --------
    pandas.DataFrame
        Flattened fitness data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # Check for common wrapper keys
        if 'data' in data:
            data = data['data']
        elif 'records' in data:
            data = data['records']
        elif 'entries' in data:
            data = data['entries']
    
    # Flatten nested JSON structures
    if isinstance(data, list):
        df = pd.json_normalize(data)
    else:
        df = pd.json_normalize([data])
    
    return df


def validate_fitness_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Quick validation check for fitness data
    Returns a validation report similar to DataValidator but simpler
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Fitness data to validate
    
    Returns:
    --------
    dict
        Validation report with issues and statistics
    """
    report = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check for timestamp column
    potential_timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Date', 'Time']
    has_timestamp = any(col in df.columns for col in potential_timestamp_cols)
    
    if not has_timestamp:
        report['issues'].append("No timestamp column found")
        report['is_valid'] = False
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        total_missing = missing_counts.sum()
        report['warnings'].append(f"Total missing values: {total_missing}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        report['warnings'].append(f"Found {duplicates} duplicate rows")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        report['warnings'].append("No numeric columns found")
    
    # Statistics
    report['statistics'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(duplicates),
        'numeric_columns': len(numeric_cols),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    
    return report


def detect_timestamp_column(df: pd.DataFrame) -> str:
    """
    Automatically detect the timestamp column in a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    str
        Name of detected timestamp column
    """
    potential_names = ['timestamp', 'date', 'datetime', 'time', 'Date', 'Time', 'DateTime', 'Timestamp']
    
    # First check for exact matches
    for col in potential_names:
        if col in df.columns:
            return col
    
    # Check for case-insensitive matches
    for col in df.columns:
        if col.lower() in [name.lower() for name in potential_names]:
            return col
    
    # Try to detect datetime-like columns by attempting conversion
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(10))
            return col
        except:
            continue
    
    raise ValueError("No timestamp column detected in the dataframe")


def detect_and_normalize_timestamps(
    df: pd.DataFrame, 
    timestamp_column: str = None,
    user_location: Optional[str] = None,
    target_timezone: str = 'UTC'
) -> pd.DataFrame:
    """
    Detect timezone and normalize timestamps to target timezone
    Complements the TimeAligner class for standalone use
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw fitness data with timestamp column
    timestamp_column : str, optional
        Name of timestamp column (auto-detected if None)
    user_location : str, optional
        User's primary location (e.g., 'New York', 'London')
    target_timezone : str
        Target timezone for normalization (default: 'UTC')
    
    Returns:
    --------
    pandas.DataFrame
        Data with normalized timestamps
    """
    df = df.copy()
    
    # Auto-detect timestamp column if not provided
    if timestamp_column is None:
        timestamp_column = detect_timestamp_column(df)
    
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found")
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce', utc=True)
    
    # Detect source timezone if not already timezone-aware
    if df[timestamp_column].dt.tz is None:
        source_tz = detect_timezone(df[timestamp_column], user_location)
        # Localize timestamps to source timezone
        df[timestamp_column] = df[timestamp_column].dt.tz_localize(
            source_tz, 
            ambiguous='infer', 
            nonexistent='shift_forward'
        )
    
    # Convert to target timezone
    df[timestamp_column] = df[timestamp_column].dt.tz_convert(target_timezone)
    
    # Add derived time-based columns
    df['hour_of_day'] = df[timestamp_column].dt.hour
    df['day_of_week'] = df[timestamp_column].dt.day_name()
    df['date'] = df[timestamp_column].dt.date
    df['is_weekend'] = df[timestamp_column].dt.dayofweek.isin([5, 6])
    
    return df


def detect_timezone(timestamps: pd.Series, user_location: Optional[str] = None) -> str:
    """
    Detect timezone from timestamp patterns or user location
    
    Parameters:
    -----------
    timestamps : pd.Series
        Series of timestamps
    user_location : str, optional
        User's location string
    
    Returns:
    --------
    str
        Detected timezone string
    """
    # Location to timezone mapping
    location_tz_map = {
        'new york': 'America/New_York',
        'london': 'Europe/London',
        'tokyo': 'Asia/Tokyo',
        'los angeles': 'America/Los_Angeles',
        'chicago': 'America/Chicago',
        'paris': 'Europe/Paris',
        'sydney': 'Australia/Sydney',
        'mumbai': 'Asia/Kolkata',
        'delhi': 'Asia/Kolkata',
        'dubai': 'Asia/Dubai',
        'singapore': 'Asia/Singapore',
        'hong kong': 'Asia/Hong_Kong',
        'toronto': 'America/Toronto'
    }
    
    if user_location:
        location_lower = user_location.lower()
        for key, tz in location_tz_map.items():
            if key in location_lower:
                return tz
    
    # If timestamps already have timezone info
    if hasattr(timestamps, 'dt') and timestamps.dt.tz is not None:
        return str(timestamps.dt.tz)
    
    # Default to UTC if cannot detect
    return 'UTC'


def handle_daylight_saving_time(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """
    Handle daylight saving time transitions
    Mark records affected by DST
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with timestamps
    timestamp_column : str
        Name of timestamp column
    
    Returns:
    --------
    pandas.DataFrame
        Data with DST information added
    """
    df = df.copy()
    
    # Check if timestamps are timezone-aware
    if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        if df[timestamp_column].dt.tz is not None:
            # Mark DST periods
            df['is_dst'] = df[timestamp_column].dt.dst.notna()
            df['dst_offset'] = df[timestamp_column].dt.dst.dt.total_seconds() / 3600
        else:
            df['is_dst'] = False
            df['dst_offset'] = 0
    
    return df


def convert_file_format(input_path: str, output_path: str, output_format: str) -> None:
    """
    Convert fitness data between different file formats
    
    Parameters:
    -----------
    input_path : str
        Path to input file
    output_path : str
        Path to output file
    output_format : str
        Target format ('csv', 'json', 'excel', 'parquet')
    """
    # Load data
    df = load_fitness_data(input_path)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert and save
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'json':
        df.to_json(output_path, orient='records', indent=2, date_format='iso')
    elif output_format in ['xlsx', 'excel']:
        df.to_excel(output_path, index=False, engine='openpyxl')
    elif output_format == 'parquet':
        df.to_parquet(output_path, index=False, engine='pyarrow')
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive data quality assessment
    Returns detailed quality metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to assess
    
    Returns:
    --------
    dict
        Quality assessment report
    """
    issues = {
        'completeness': {},
        'consistency': {},
        'accuracy': {},
        'timeliness': {}
    }
    
    # Completeness checks
    missing_by_col = df.isnull().sum()
    issues['completeness'] = {
        'total_missing': int(missing_by_col.sum()),
        'missing_percentage': round(missing_by_col.sum() / (len(df) * len(df.columns)) * 100, 2),
        'columns_with_missing': missing_by_col[missing_by_col > 0].to_dict()
    }
    
    # Consistency checks
    issues['consistency'] = {
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_percentage': round(df.duplicated().sum() / len(df) * 100, 2)
    }
    
    # Accuracy checks (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_counts[col] = int(outliers)
    
    issues['accuracy'] = {
        'outlier_counts': outlier_counts,
        'total_outliers': sum(outlier_counts.values())
    }
    
    # Timeliness checks (if timestamp column exists)
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if timestamp_cols:
        ts_col = timestamp_cols[0]
        try:
            ts_series = pd.to_datetime(df[ts_col], errors='coerce')
            now = pd.Timestamp.now(tz='UTC')
            future_dates = (ts_series > now).sum()
            very_old_dates = (ts_series < pd.Timestamp('2010-01-01', tz='UTC')).sum()
            
            issues['timeliness'] = {
                'future_dates': int(future_dates),
                'very_old_dates': int(very_old_dates),
                'date_range': f"{ts_series.min()} to {ts_series.max()}"
            }
        except:
            issues['timeliness'] = {'error': 'Could not parse timestamps'}
    
    return issues


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores
    Maps common variations to standard names
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with standardized column names
    """
    df = df.copy()
    
    # Common mappings
    column_mappings = {
        'time': 'timestamp',
        'date': 'timestamp',
        'datetime': 'timestamp',
        'hr': 'heart_rate',
        'heartrate': 'heart_rate',
        'bpm': 'heart_rate',
        'pulse': 'heart_rate',
        'steps': 'step_count',
        'stepcount': 'step_count',
        'sleep': 'sleep_stage',
        'stage': 'sleep_stage',
        'duration': 'duration_minutes',
        'minutes': 'duration_minutes'
    }
    
    # Standardize to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Apply mappings
    df = df.rename(columns=column_mappings)
    
    return df


def resample_time_series(
    df: pd.DataFrame, 
    timestamp_col: str = 'timestamp',
    freq: str = '1min',
    agg_method: str = 'mean'
) -> pd.DataFrame:
    """
    Resample time series data to uniform frequency
    Simplified version of TimeAligner for quick resampling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input time series data
    timestamp_col : str
        Name of timestamp column
    freq : str
        Target frequency (e.g., '1min', '5min', '1H')
    agg_method : str
        Aggregation method ('mean', 'sum', 'median', 'first', 'last')
    
    Returns:
    --------
    pandas.DataFrame
        Resampled dataframe
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set timestamp as index
    df = df.set_index(timestamp_col).sort_index()
    
    # Resample based on aggregation method
    if agg_method == 'mean':
        df_resampled = df.resample(freq).mean()
    elif agg_method == 'sum':
        df_resampled = df.resample(freq).sum()
    elif agg_method == 'median':
        df_resampled = df.resample(freq).median()
    elif agg_method == 'first':
        df_resampled = df.resample(freq).first()
    elif agg_method == 'last':
        df_resampled = df.resample(freq).last()
    else:
        df_resampled = df.resample(freq).mean()
    
    # Reset index
    df_resampled = df_resampled.reset_index()
    
    return df_resampled