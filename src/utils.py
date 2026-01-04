"""
Utility Functions for FitPulse Health Anomaly Detection
Helper functions to complement the main preprocessing pipeline
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import streamlit as st
from datetime import datetime, timedelta


def get_file_info(file_path: str) -> Dict[str, any]:
    """
    Get detailed information about a file
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    
    Returns:
    --------
    dict
        File information including size, type, modified time
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    stats = file_path.stat()
    
    return {
        'name': file_path.name,
        'size_bytes': stats.st_size,
        'size_mb': round(stats.st_size / (1024 * 1024), 2),
        'size_kb': round(stats.st_size / 1024, 2),
        'extension': file_path.suffix,
        'modified_time': pd.Timestamp.fromtimestamp(stats.st_mtime),
        'created_time': pd.Timestamp.fromtimestamp(stats.st_ctime)
    }


def compare_file_formats(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare different file formats for the same data
    Useful for Task 2 format comparison analysis
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with format names as keys and dataframes as values
    
    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics
    """
    comparisons = []
    
    for format_name, df in data_dict.items():
        # Measure loading/copy speed
        start = time.time()
        _ = df.copy()
        load_time = time.time() - start
        
        # Calculate memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        comparisons.append({
            'Format': format_name,
            'Rows': f"{len(df):,}",
            'Columns': len(df.columns),
            'Memory (MB)': round(memory_usage, 2),
            'Load Time (s)': round(load_time, 4),
            'Avg Row Size (KB)': round((memory_usage * 1024) / len(df), 2) if len(df) > 0 else 0
        })
    
    return pd.DataFrame(comparisons)


def create_format_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Create visualization comparing file formats
    Interactive chart for Task 2
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison data from compare_file_formats
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Comparison chart with dual axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            name='Memory Usage (MB)',
            x=comparison_df['Format'],
            y=comparison_df['Memory (MB)'],
            marker_color='#667eea',
            text=comparison_df['Memory (MB)'],
            textposition='outside'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            name='Load Time (s)',
            x=comparison_df['Format'],
            y=comparison_df['Load Time (s)'],
            marker_color='#f093fb',
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=3)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Format")
    fig.update_yaxes(title_text="Memory Usage (MB)", secondary_y=False)
    fig.update_yaxes(title_text="Load Time (seconds)", secondary_y=True)
    
    fig.update_layout(
        title='File Format Performance Comparison',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_timestamp_heatmap(df: pd.DataFrame, timestamp_column: str) -> go.Figure:
    """
    Create 24-hour heatmap showing data coverage by day and hour
    For Task 3 timestamp validation dashboard
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with timestamps
    timestamp_column : str
        Name of timestamp column
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap
    """
    df_copy = df.copy()
    
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_column]):
        df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
    
    # Extract hour and day of week
    df_copy['hour'] = df_copy[timestamp_column].dt.hour
    df_copy['day_of_week'] = df_copy[timestamp_column].dt.day_name()
    
    # Create pivot table for heatmap
    heatmap_data = df_copy.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([day for day in day_order if day in heatmap_pivot.index])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Viridis',
        text=heatmap_pivot.values.astype(int),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title="Record Count")
    ))
    
    fig.update_layout(
        title='Data Coverage: 24-Hour Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=450,
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    return fig


def create_timezone_distribution(df: pd.DataFrame, timestamp_column: str) -> go.Figure:
    """
    Visualize timezone distribution and hourly patterns
    For Task 3 validation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with timestamps
    timestamp_column : str
        Name of timestamp column
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Distribution chart
    """
    df_copy = df.copy()
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_column]):
        df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
    
    # Get timezone info
    if df_copy[timestamp_column].dt.tz is not None:
        tz_name = str(df_copy[timestamp_column].dt.tz)
    else:
        tz_name = "No timezone (naive)"
    
    # Create hourly distribution
    df_copy['hour'] = df_copy[timestamp_column].dt.hour
    hour_dist = df_copy['hour'].value_counts().sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hour_dist.index,
        y=hour_dist.values,
        marker_color='#667eea',
        text=hour_dist.values,
        textposition='outside',
        hovertemplate='Hour: %{x}<br>Records: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Data Distribution by Hour<br><sub>Timezone: {tz_name}</sub>',
        xaxis_title='Hour of Day',
        yaxis_title='Record Count',
        height=400,
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        showlegend=False
    )
    
    return fig


def create_before_after_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    timestamp_column: str
) -> go.Figure:
    """
    Visualize before/after timestamp conversion
    Shows impact of timezone normalization (Task 3)
    
    Parameters:
    -----------
    df_before : pd.DataFrame
        Data before conversion
    df_after : pd.DataFrame
        Data after conversion
    timestamp_column : str
        Name of timestamp column
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Comparison visualization
    """
    # Sample data for clarity
    sample_size = min(100, len(df_before))
    
    fig = go.Figure()
    
    # Before conversion
    fig.add_trace(go.Scatter(
        x=list(range(sample_size)),
        y=pd.to_datetime(df_before[timestamp_column]).head(sample_size).dt.hour,
        mode='markers',
        name='Before Conversion',
        marker=dict(color='#f093fb', size=10, opacity=0.7),
        hovertemplate='Index: %{x}<br>Hour: %{y}<extra></extra>'
    ))
    
    # After conversion
    fig.add_trace(go.Scatter(
        x=list(range(sample_size)),
        y=pd.to_datetime(df_after[timestamp_column]).head(sample_size).dt.hour,
        mode='markers',
        name='After Conversion',
        marker=dict(color='#667eea', size=10, opacity=0.7),
        hovertemplate='Index: %{x}<br>Hour: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Timestamp Conversion Impact: Before vs After',
        xaxis_title='Sample Index',
        yaxis_title='Hour of Day',
        height=450,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def display_dataset_info(df: pd.DataFrame, data_type: str = "Dataset") -> None:
    """
    Display comprehensive dataset information in Streamlit
    Enhanced version with more metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to display info for
    data_type : str
        Name/type of dataset for display
    """
    st.subheader(f"📊 {data_type} Overview")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        missing_pct = (missing / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
        st.metric("Missing Values", f"{missing:,}", delta=f"{missing_pct:.1f}%")
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")
    
    # Detailed column information
    st.subheader("📋 Column Information")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': f"{df[col].notna().sum():,}",
            'Null': f"{df[col].isna().sum():,}",
            'Unique': f"{df[col].nunique():,}",
            'Sample': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True, hide_index=True)


def generate_sample_data(data_type: str = 'heart_rate', rows: int = 100) -> pd.DataFrame:
    """
    Generate realistic sample fitness data for testing
    Matches the patterns expected by FitPulse preprocessing
    
    Parameters:
    -----------
    data_type : str
        Type of data ('heart_rate', 'steps', 'sleep')
    rows : int
        Number of rows to generate
    
    Returns:
    --------
    pd.DataFrame
        Sample fitness data
    """
    base_time = datetime.now()
    timestamps = [base_time + timedelta(minutes=i*5) for i in range(rows)]
    
    if data_type == 'heart_rate':
        # Simulate realistic heart rate with circadian rhythm
        hr_values = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Base rate varies by time of day
            if 0 <= hour < 6:
                base = 60  # Sleep/rest
            elif 6 <= hour < 12:
                base = 75  # Morning
            elif 12 <= hour < 18:
                base = 80  # Afternoon/active
            else:
                base = 70  # Evening
            
            # Add variation and slight trend
            hr = base + np.random.normal(0, 5) + 5 * np.sin(2 * np.pi * i / 120)
            hr_values.append(max(50, min(120, hr)))
        
        data = {
            'timestamp': timestamps,
            'heart_rate': hr_values,
            'activity_type': np.random.choice(['resting', 'walking', 'light_activity', 'moderate_activity'], rows, p=[0.4, 0.3, 0.2, 0.1])
        }
    
    elif data_type == 'steps':
        # Simulate step count data
        step_values = []
        for ts in timestamps:
            hour = ts.hour
            # More steps during active hours
            if 7 <= hour < 20:
                steps = np.random.randint(50, 500)
            else:
                steps = np.random.randint(0, 50)
            step_values.append(steps)
        
        data = {
            'timestamp': timestamps,
            'step_count': step_values,
            'distance_km': [s * 0.0008 for s in step_values],  # Approximate distance
            'calories': [s * 0.04 for s in step_values]  # Approximate calories
        }
    
    elif data_type == 'sleep':
        # Simulate sleep stage data
        data = {
            'timestamp': timestamps,
            'sleep_stage': np.random.choice(['deep', 'light', 'rem', 'awake'], rows, p=[0.2, 0.4, 0.25, 0.15]),
            'heart_rate': np.random.randint(45, 75, rows),
            'duration_minutes': np.random.randint(5, 90, rows)
        }
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return pd.DataFrame(data)


def export_report(report_data: Dict, output_path: str, format: str = 'json') -> None:
    """
    Export validation/processing report to file
    
    Parameters:
    -----------
    report_data : dict
        Report data to export
    output_path : str
        Path to output file
    format : str
        Export format ('json', 'txt', 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        import json
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    elif format == 'txt':
        with open(output_path, 'w') as f:
            f.write("FitPulse Processing Report\n")
            f.write("=" * 50 + "\n\n")
            for key, value in report_data.items():
                f.write(f"{key}: {value}\n")
    
    elif format == 'csv':
        # Flatten nested dict for CSV export
        flat_data = {}
        for key, value in report_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_data[f"{key}_{sub_key}"] = sub_value
            else:
                flat_data[key] = value
        
        df = pd.DataFrame([flat_data])
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


def safe_file_upload(uploaded_file, save_directory: str = "data/raw") -> str:
    """
    Safely handle file upload and save to directory
    Used with Streamlit file uploader
    
    Parameters:
    -----------
    uploaded_file : streamlit.UploadedFile
        File uploaded via Streamlit
    save_directory : str
        Directory to save file to
    
    Returns:
    --------
    str
        Path to saved file
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique filename if file already exists
    file_path = save_dir / uploaded_file.name
    counter = 1
    while file_path.exists():
        name_parts = uploaded_file.name.rsplit('.', 1)
        if len(name_parts) == 2:
            file_path = save_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
        else:
            file_path = save_dir / f"{uploaded_file.name}_{counter}"
        counter += 1
    
    # Write file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate overall data quality score (0-100)
    Based on completeness, consistency, and validity
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to assess
    
    Returns:
    --------
    float
        Quality score between 0 and 100
    """
    scores = []
    
    # Completeness (40 points): Percentage of non-null values
    completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    scores.append(completeness * 40)
    
    # Consistency (30 points): Based on duplicates
    if len(df) > 0:
        duplicate_rate = df.duplicated().sum() / len(df)
        consistency = (1 - duplicate_rate) * 30
    else:
        consistency = 30
    scores.append(consistency)
    
    # Validity (30 points): Based on data types and outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        outlier_rates = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_rates.append(outliers / len(df))
        
        avg_outlier_rate = np.mean(outlier_rates)
        validity = (1 - avg_outlier_rate) * 30
    else:
        validity = 30
    scores.append(validity)
    
    return round(sum(scores), 2)


def create_data_quality_gauge(quality_score: float) -> go.Figure:
    """
    Create a gauge chart for data quality score
    
    Parameters:
    -----------
    quality_score : float
        Quality score between 0 and 100
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=quality_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Quality Score", 'font': {'size': 24}},
        delta={'reference': 75, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#ffffcc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig