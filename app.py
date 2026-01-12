import setuptools  # Ensure pkg_resources is available for TSFresh

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import warnings
import io
from scipy import stats
from scipy.signal import find_peaks
import hashlib

# Import anomaly detection modules
from anomaly_detection import (
    ThresholdAnomalyDetector,
    ResidualAnomalyDetector,
    ClusterAnomalyDetector,
    create_sample_data_with_anomalies
)
from anomaly_visualization import AnomalyVisualizer

from src.utils import (
    compare_file_formats, 
    create_format_comparison_chart,
    create_timestamp_heatmap,
    create_timezone_distribution
)

warnings.filterwarnings('ignore')

# Configure Streamlit page - MUST be first Streamlit command
st.set_page_config(
    page_title="FitPulse - Module 1: Data Preprocessing",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_THEME = {
    'template': 'plotly_white',
    'font': dict(family='Inter, sans-serif', size=12, color='#495057'),
    'title_font': dict(size=18, color='#1e3a8a', family='Inter, sans-serif'),
    'colorway': ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#fa709a', '#fee140'],
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'margin': dict(l=60, r=60, t=80, b=60),
    'hovermode': 'x unified',
    'hoverlabel': dict(
        bgcolor='white',
        font_size=13,
        font_family='Inter, sans-serif',
        bordercolor='#667eea'
    )
}

def apply_chart_theme(fig):
    fig.update_layout(**CHART_THEME)
    fig.update_layout(
        title_font_size=18,
        title_font_color='#1e3a8a',
        title_font_family='Inter, sans-serif',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e0e0e0',
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.05)',
            showline=True,
            linewidth=2,
            linecolor='#e0e0e0'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.05)',
            showline=True,
            linewidth=2,
            linecolor='#e0e0e0'
        )
    )
    return fig

def style_dataframe(df):
    return df.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border-color': '#dee2e6',
        'font-family': 'Inter, sans-serif',
        'font-size': '13px'
    }).set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background', 'linear-gradient(135deg, #667eea, #764ba2)'),
            ('color', 'white'),
            ('font-weight', '700'),
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('padding', '12px')
        ]},
        {'selector': 'tbody tr:hover', 'props': [
            ('background-color', '#e7f1ff')
        ]},
        {'selector': 'tbody td', 'props': [
            ('padding', '10px'),
            ('text-align', 'center')
        ]}
    ])


class DataLoader:
    """Enterprise-grade data loading with comprehensive format support"""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv'],
        'json': ['.json'],
        'excel': ['.xlsx', '.xls']
    }
    
    DATA_TYPE_SCHEMAS = {
        'heart_rate': {
            'required': ['timestamp', 'heart_rate'],
            'optional': ['activity_type', 'confidence'],
            'constraints': {'heart_rate': (30, 220)}
        },
        'sleep': {
            'required': ['date', 'duration_minutes'],
            'optional': ['quality_score', 'sleep_stage'],
            'constraints': {'duration_minutes': (0, 1440)}
        },
        'steps': {
            'required': ['timestamp', 'step_count'],
            'optional': ['distance_km', 'calories'],
            'constraints': {'step_count': (0, 100000)}
        }
    }
    
    def __init__(self):
        self.loaded_data = {}
        self.load_metadata = {}
        
    def load_files(self, uploaded_files: List) -> Dict[str, pd.DataFrame]:
        """Load and parse multiple file formats with intelligent type detection"""
        results = {}
        
        for file in uploaded_files:
            try:
               
                data_type = self._detect_data_type(file.name)
                
                # Determine date column name based on data type
                date_column = 'date' if data_type == 'sleep' else 'timestamp'
              
                if file.name.endswith('.csv'):
                    # Ensure StringIO objects are at the beginning
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    df = pd.read_csv(file, parse_dates=[date_column], encoding='utf-8-sig')
                elif file.name.endswith('.json'):
                    # Ensure StringIO objects are at the beginning
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    df = self._parse_json(file)
                elif file.name.endswith(('.xlsx', '.xls')):
                    # For Excel files, we need to handle StringIO differently
                    # This is a simplified approach - in practice, Excel files from StringIO might need special handling
                    df = pd.read_excel(file, parse_dates=[date_column])
                else:
                    st.warning(f"⚠️ Unsupported format: {file.name}")
                    continue
                
              
                if self._validate_schema(df, data_type):
                    results[data_type] = df
                    self.load_metadata[data_type] = {
                        'filename': file.name,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'loaded_at': datetime.now().isoformat()
                    }
                    st.success(f"✅ {file.name}: {len(df):,} records loaded")
                else:
                    st.error(f"❌ {file.name}: Schema validation failed")
                    
            except Exception as e:
                st.error(f"❌ Error loading {file.name}: {str(e)}")
                
        return results
    
    def _detect_data_type(self, filename: str) -> str:
        """Intelligent data type detection from filename"""
        name_lower = filename.lower()
        
        if any(kw in name_lower for kw in ['heart', 'hr', 'pulse', 'bpm']):
            return 'heart_rate'
        elif any(kw in name_lower for kw in ['sleep', 'rest', 'bed']):
            return 'sleep'
        elif any(kw in name_lower for kw in ['step', 'walk', 'activity']):
            return 'steps'
        else:
            return 'unknown'
    
    def _parse_json(self, file) -> pd.DataFrame:
        """Parse JSON with nested structure support"""
        # Ensure StringIO objects are at the beginning
        if hasattr(file, 'seek'):
            file.seek(0)
        data = json.load(file)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                return pd.DataFrame(data['data'])
            elif 'records' in data:
                return pd.DataFrame(data['records'])
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
    
    def _validate_schema(self, df: pd.DataFrame, data_type: str) -> bool:
        """Validate dataframe against expected schema with flexible matching"""
        if data_type == 'unknown':
            st.warning(f"⚠️ Could not detect data type from filename. Attempting to process anyway...")
            return True 
        
        if data_type not in self.DATA_TYPE_SCHEMAS:
            st.warning(f"⚠️ Unknown data type: {data_type}. Processing with default settings...")
            return True
        
        schema = self.DATA_TYPE_SCHEMAS[data_type]
        df_cols_lower = [col.lower().strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        
       
        missing_cols = []
        for req_col in schema['required']:
           
            if req_col not in df_cols_lower:
               
                found = any(req_col in col or col in req_col for col in df_cols_lower)
                if not found:
                    missing_cols.append(req_col)
        
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            st.info(f"📋 Available columns: {', '.join(df.columns)}")
            st.info(f"💡 Tip: Ensure your CSV has columns like: {', '.join(schema['required'])}")
            return False
        
        return True
    
    def load_fitness_data(self, file_path: str, file_type: str = 'auto') -> pd.DataFrame:
        """
        Load fitness data from various file formats
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'json', 'excel', or 'auto' for auto-detection)
        
        Returns:
            DataFrame with loaded data
        """
        import os
        
       
        if file_type == 'auto':
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.json':
                file_type = 'json'
            elif ext in ['.xlsx', '.xls']:
                file_type = 'excel'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        
        try:
            # Determine data type and date column
            data_type = self._detect_data_type(os.path.basename(file_path))
            date_column = 'date' if data_type == 'sleep' else 'timestamp'
            
            if file_type == 'csv':
                df = pd.read_csv(file_path, parse_dates=[date_column])
            elif file_type == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data if isinstance(data, list) else data.get('data', [data]))
            elif file_type == 'excel':
                df = pd.read_excel(file_path, parse_dates=[date_column])
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")
    
    def analyze_formats(self, sample_data: pd.DataFrame) -> Dict:
        """
        Compare different file formats for performance and size
        
        Args:
            sample_data: DataFrame to test with different formats
        
        Returns:
            Dictionary with comparison results
        """
        import time
        import os
        import tempfile
        
        results = {}
        
      
        for fmt in ['csv', 'json', 'excel']:
            try:
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{fmt}') as tmp:
                    tmp_path = tmp.name
                
               
                start = time.time()
                if fmt == 'csv':
                    sample_data.to_csv(tmp_path, index=False)
                elif fmt == 'json':
                    sample_data.to_json(tmp_path, orient='records', date_format='iso')
                elif fmt == 'excel':
                    sample_data.to_excel(tmp_path, index=False, engine='openpyxl')
                save_time = time.time() - start
                
               
                file_size = os.path.getsize(tmp_path) / 1024  # KB
                
               
                start = time.time()
                if fmt == 'csv':
                    _ = pd.read_csv(tmp_path)
                elif fmt == 'json':
                    _ = pd.read_json(tmp_path)
                elif fmt == 'excel':
                    _ = pd.read_excel(tmp_path)
                load_time = time.time() - start
                
                results[fmt] = {
                    'save_time': save_time,
                    'load_time': load_time,
                    'file_size_kb': file_size,
                    'total_time': save_time + load_time
                }
                
               
                os.unlink(tmp_path)
                
            except Exception as e:
                results[fmt] = {'error': str(e)}
        
     
        self._plot_format_comparison(results)
        
        return results
    
    def _plot_format_comparison(self, results: Dict):
        """Create visualization comparing file formats"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        formats = list(results.keys())
        save_times = [results[f].get('save_time', 0) for f in formats]
        load_times = [results[f].get('load_time', 0) for f in formats]
        file_sizes = [results[f].get('file_size_kb', 0) for f in formats]
        
      
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Save Time (s)', 'Load Time (s)', 'File Size (KB)'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
       
        fig.add_trace(
            go.Bar(x=formats, y=save_times, name='Save Time',
                  marker=dict(color='#667eea')),
            row=1, col=1
        )
     
        fig.add_trace(
            go.Bar(x=formats, y=load_times, name='Load Time',
                  marker=dict(color='#f093fb')),
            row=1, col=2
        )
        
      
        fig.add_trace(
            go.Bar(x=formats, y=file_sizes, name='File Size',
                  marker=dict(color='#4facfe')),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text="📊 File Format Performance Comparison",
            showlegend=False,
            height=450,
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


class DataValidator:
    """Advanced data validation with statistical quality checks"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Comprehensive validation pipeline"""
        report = {
            'original_rows': len(df),
            'data_type': data_type,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        df_clean = df.copy()
       
        df_clean = self._standardize_columns(df_clean)
        
      
        df_clean, ts_issues = self._validate_timestamps(df_clean)
        report['issues'].extend(ts_issues)
        
        
        df_clean, range_issues = self._validate_ranges(df_clean, data_type)
        report['issues'].extend(range_issues)
        
      
        df_clean, missing_info = self._handle_missing(df_clean, data_type)
        report['metrics']['missing_handled'] = missing_info
        
        
        df_clean, outlier_info = self._detect_outliers(df_clean)
        report['metrics']['outliers_found'] = outlier_info
        
        
        df_clean, dup_count = self._remove_duplicates(df_clean)
        report['metrics']['duplicates_removed'] = dup_count
        
 
        report['metrics']['quality_score'] = self._calculate_quality_score(df_clean, report)
        
        report['final_rows'] = len(df_clean)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
        
        return df_clean, report
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names with comprehensive mapping"""
        mapping = {
            'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
            'hr': 'heart_rate', 'heartrate': 'heart_rate', 'bpm': 'heart_rate',
            'steps': 'step_count', 'stepcount': 'step_count',
            'sleep': 'sleep_stage', 'stage': 'sleep_stage',
            'duration': 'duration_minutes', 'minutes': 'duration_minutes'
        }
        
        df = df.rename(columns=mapping)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced timestamp validation with timezone handling"""
        issues = []
        
        if 'timestamp' not in df.columns:
            issues.append("CRITICAL: No timestamp column found")
            return df, issues
        
        try:
      
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            
           
            invalid_count = df['timestamp'].isna().sum()
            if invalid_count > 0:
                issues.append(f"Failed to parse {invalid_count} timestamps ({invalid_count/len(df)*100:.1f}%)")
            
         
            future_dates = (df['timestamp'] > pd.Timestamp.now(tz='UTC')).sum()
            if future_dates > 0:
                issues.append(f"Found {future_dates} future timestamps")
                df = df[df['timestamp'] <= pd.Timestamp.now(tz='UTC')]
            
           
            old_dates = (df['timestamp'] < pd.Timestamp('2010-01-01', tz='UTC')).sum()
            if old_dates > 0:
                issues.append(f"Found {old_dates} timestamps before 2010")
                df = df[df['timestamp'] >= pd.Timestamp('2010-01-01', tz='UTC')]
            
        except Exception as e:
            issues.append(f"Timestamp processing error: {str(e)}")
        
        return df, issues
    
    def _validate_ranges(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """Validate numeric values against physiological/realistic ranges"""
        issues = []
        
        range_checks = {
            'heart_rate': (30, 220, 'Heart rate'),
            'step_count': (0, 100000, 'Step count'),
            'duration_minutes': (0, 1440, 'Duration')
        }
        
        for col, (min_val, max_val, label) in range_checks.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    issues.append(f"{label}: {out_of_range} values outside range [{min_val}, {max_val}]")
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
        
        return df, issues
    
    def _handle_missing(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Intelligent missing value handling with multiple strategies"""
        info = {
            'before': df.isnull().sum().to_dict(),
            'strategy': {},
            'after': {}
        }
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            if missing_pct > 0:
                if col == 'timestamp':
                   
                    df = df.dropna(subset=['timestamp'])
                    info['strategy'][col] = 'dropped'
                    
                elif col in ['heart_rate', 'step_count']:
                   
                    df[col] = df[col].interpolate(method='linear', limit=10)
                    df[col] = df[col].fillna(method='bfill', limit=5)
                    info['strategy'][col] = 'interpolation + backfill'
                    
                elif col == 'sleep_stage':
                  
                    df[col] = df[col].fillna(method='ffill')
                    info['strategy'][col] = 'forward fill'
                    
                else:
                    
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                        info['strategy'][col] = 'median'
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
                        info['strategy'][col] = 'mode'
        
        info['after'] = df.isnull().sum().to_dict()
        return df, info
    
    def _detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Statistical outlier detection using IQR and Z-score methods"""
        outlier_info = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'timestamp':
              
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                df[f'{col}_is_outlier'] = outliers
                
                outlier_info[col] = {
                    'count': outliers.sum(),
                    'percentage': f"{outliers.sum() / len(df) * 100:.2f}%",
                    'bounds': (lower_bound, upper_bound)
                }
        
        return df, outlier_info
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate records based on timestamp"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        duplicates_removed = initial_count - len(df)
        return df, duplicates_removed
    
    def _calculate_quality_score(self, df: pd.DataFrame, report: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        scores = []
        
    
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        scores.append(completeness * 0.4)
        
       
        issue_penalty = min(len(report['issues']) * 0.05, 0.3)
        validity = max(0, 0.3 - issue_penalty)
        scores.append(validity)
        
      
        outlier_cols = [col for col in df.columns if col.endswith('_is_outlier')]
        if outlier_cols:
            outlier_rate = df[outlier_cols].sum().sum() / (len(df) * len(outlier_cols))
            consistency = (1 - outlier_rate) * 0.3
        else:
            consistency = 0.3
        scores.append(consistency)
        
        return round(sum(scores) * 100, 2)


class TimeAligner:
    """Advanced time series alignment with adaptive resampling"""
    
    FREQUENCIES = {
        '30 sec': '30S',
        '1 min': '1T',
        '5 min': '5T',
        '15 min': '15T',
        '30 min': '30T',
        '1 hour': '1H'
    }
    
    def align(self, df: pd.DataFrame, data_type: str, 
              target_freq: str = '1 min', fill_method: str = 'interpolate') -> Tuple[pd.DataFrame, Dict]:
        """Align timestamps and resample to uniform frequency"""
        report = {
            'data_type': data_type,
            'original_rows': len(df),
            'original_frequency': 'unknown',
            'target_frequency': target_freq,
            'success': False
        }
        
        try:
            # Special handling for sleep data - it's already daily, don't resample
            if data_type == 'sleep':
                report['original_frequency'] = 'daily'
                report['resampled_rows'] = len(df)
                report['time_span'] = str(df['timestamp'].max() - df['timestamp'].min()) if 'timestamp' in df.columns else 'unknown'
                report['success'] = True
                return df, report
          
            df = df.set_index('timestamp').sort_index()
         
            report['original_frequency'] = self._detect_frequency(df)
            
            # Check if resampling is actually needed
            original_freq_minutes = self._freq_to_minutes(report['original_frequency'])
            target_freq_minutes = self._freq_to_minutes(target_freq)
            
            # Calculate expected record count after resampling
            if original_freq_minutes is not None and target_freq_minutes is not None:
                time_span_minutes = (df.index.max() - df.index.min()).total_seconds() / 60
                expected_records = time_span_minutes / target_freq_minutes
                
                # Don't resample if it would increase records by more than 5x
                if expected_records > len(df) * 5:
                    report['resampled_rows'] = len(df)
                    report['time_span'] = str(df.index.max() - df.index.min())
                    report['success'] = True
                    return df.reset_index(), report
            
            # Proceed with resampling
            df_resampled = self._smart_resample(df, data_type, self.FREQUENCIES[target_freq])
            
        
            df_filled = self._fill_gaps(df_resampled, fill_method)
         
            df_final = df_filled.reset_index()
            
            report['resampled_rows'] = len(df_final)
            report['time_span'] = str(df_final['timestamp'].max() - df_final['timestamp'].min())
            report['success'] = True
            
            return df_final, report
            
        except Exception as e:
            report['error'] = str(e)
            return df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df, report
    
    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect the dominant frequency in the time series"""
        if len(df) < 2:
            return "insufficient_data"
        
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        seconds = median_diff.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)}sec"
        elif seconds < 3600:
            return f"{int(seconds/60)}min"
        else:
            return f"{seconds/3600:.1f}hour"
    
    def _smart_resample(self, df: pd.DataFrame, data_type: str, freq: str) -> pd.DataFrame:
        """Data-type aware resampling with appropriate aggregation"""
        resampled_dict = {}
        
        for col in df.columns:
            if col.endswith('_is_outlier'):
                resampled_dict[col] = df[col].resample(freq).max()
            elif col == 'heart_rate':
                resampled_dict[col] = df[col].resample(freq).mean()
            elif col == 'step_count':
                resampled_dict[col] = df[col].resample(freq).sum()
            elif col == 'duration_minutes':
                resampled_dict[col] = df[col].resample(freq).sum()
            elif col == 'sleep_stage':
                resampled_dict[col] = df[col].resample(freq).agg(
                    lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
                )
            else:
                if df[col].dtype in ['int64', 'float64']:
                    resampled_dict[col] = df[col].resample(freq).mean()
                else:
                    resampled_dict[col] = df[col].resample(freq).first()
        
        return pd.DataFrame(resampled_dict)
    
    def _freq_to_minutes(self, freq_str: str) -> Optional[int]:
        """Convert frequency string to minutes for comparison"""
        if not freq_str or freq_str == 'unknown' or freq_str == 'insufficient_data':
            return None
        
        # Handle common frequency formats
        freq_str = freq_str.lower().strip()
        
        if 'min' in freq_str:
            # Extract number before 'min'
            import re
            match = re.search(r'(\d+)', freq_str)
            if match:
                return int(match.group(1))
        elif 'sec' in freq_str:
            # Convert seconds to minutes
            import re
            match = re.search(r'(\d+)', freq_str)
            if match:
                return int(match.group(1)) / 60
        elif 'hour' in freq_str:
            # Convert hours to minutes
            import re
            match = re.search(r'(\d+)', freq_str)
            if match:
                return int(match.group(1)) * 60
        
        return None
    
    def _fill_gaps(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Fill missing values after resampling"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'interpolate':
            for col in numeric_cols:
                if not col.endswith('_is_outlier'):
                    df[col] = df[col].interpolate(method='time', limit_direction='both')
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'zero':
            df = df.fillna(0)
        
        return df


class TimezoneProcessor:
    """Multi-timezone fitness data processor with automatic detection and normalization"""
    
    def __init__(self):
        self.detected_timezones = {}
        self.dst_transitions = []
    
    def detect_and_normalize_timestamps(self, df: pd.DataFrame, user_location: str = None) -> pd.DataFrame:
        """
        Detect timezone and normalize to UTC
        
        Args:
            df: DataFrame with timestamp column
            user_location: Optional user location for timezone detection
        
        Returns:
            DataFrame with normalized timestamps
        """
        df = df.copy()
        
       
        detected_tz = self._detect_timezone(df, user_location)
        
      
        df['timestamp'] = self._handle_mixed_timezones(df['timestamp'], detected_tz)
        
        if df['timestamp'].dt.tz is None:
           
            df['timestamp'] = df['timestamp'].dt.tz_localize(detected_tz, ambiguous='infer', nonexistent='shift_forward')
        
        df['timestamp_utc'] = df['timestamp'].dt.tz_convert('UTC')
        
      
        df = self._handle_dst_transitions(df)
        
        return df
    
    def _detect_timezone(self, df: pd.DataFrame, user_location: str = None) -> str:
        """
        Automatic timezone detection from patterns
        
        Args:
            df: DataFrame with timestamp column
            user_location: Optional user location hint
        
        Returns:
            Detected timezone string
        """
        import pytz
        
        
        if user_location:
            try:
                tz = pytz.timezone(user_location)
                self.detected_timezones['user_provided'] = user_location
                return user_location
            except:
                pass
        
        if df['timestamp'].dt.tz is not None:
            detected = str(df['timestamp'].dt.tz)
            self.detected_timezones['from_data'] = detected
            return detected
       
        if 'timestamp' in df.columns:
            hours = df['timestamp'].dt.hour
            
            peak_hour = hours.mode()[0] if len(hours.mode()) > 0 else 12
            
           
            if 6 <= peak_hour <= 20:
               
                detected = 'UTC'
            else:
                detected = 'UTC'
        else:
            detected = 'UTC'
        
        self.detected_timezones['detected'] = detected
        return detected
    
    def _handle_mixed_timezones(self, timestamps: pd.Series, default_tz: str) -> pd.Series:
        """
        Handle mixed timezone data
        
        Args:
            timestamps: Series of timestamps
            default_tz: Default timezone to use
        
        Returns:
            Normalized timestamp series
        """
        import pytz
        
      
        if timestamps.dt.tz is not None:
            return timestamps
        
        
        try:
            tz = pytz.timezone(default_tz)
            return timestamps.dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
        except:
          
            return timestamps.dt.tz_localize('UTC')
    
    def _handle_dst_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle daylight saving time edge cases
        
        Args:
            df: DataFrame with timestamp columns
        
        Returns:
            DataFrame with DST transitions handled
        """
        
        if 'timestamp' in df.columns and df['timestamp'].dt.tz is not None:
           
            time_diffs = df['timestamp'].diff()
            
           
            median_diff = time_diffs.median()
            unusual_gaps = time_diffs[time_diffs > median_diff * 2]
            
            if len(unusual_gaps) > 0:
                self.dst_transitions = unusual_gaps.index.tolist()
                df['dst_transition'] = False
                df.loc[unusual_gaps.index, 'dst_transition'] = True
        
        return df
    
    def create_timezone_dashboard(self, df: pd.DataFrame):
        """
        Visualization: before/after, distribution, 24hr heatmap
        
        Args:
            df: DataFrame with timestamp data
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        st.subheader("🌍 Timezone Analysis Dashboard")
        
      
        if 'timestamp' in df.columns and 'timestamp_utc' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Original Timestamps', 'UTC Normalized Timestamps'),
                vertical_spacing=0.15
            )
            
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df.index, mode='markers',
                          name='Original', marker=dict(color='#667eea', size=4)),
                row=1, col=1
            )
            
      
            fig.add_trace(
                go.Scatter(x=df['timestamp_utc'], y=df.index, mode='markers',
                          name='UTC', marker=dict(color='#f093fb', size=4)),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
       
        col1, col2 = st.columns(2)
        
        with col1:
            if 'timestamp' in df.columns:
                hours = df['timestamp'].dt.hour
                fig = go.Figure(data=[
                    go.Histogram(x=hours, nbinsx=24, marker=dict(color='#667eea'))
                ])
                fig.update_layout(
                    title='Hour Distribution (Original)',
                    xaxis_title='Hour of Day',
                    yaxis_title='Count',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'timestamp_utc' in df.columns:
                hours_utc = df['timestamp_utc'].dt.hour
                fig = go.Figure(data=[
                    go.Histogram(x=hours_utc, nbinsx=24, marker=dict(color='#f093fb'))
                ])
                fig.update_layout(
                    title='Hour Distribution (UTC)',
                    xaxis_title='Hour of Day',
                    yaxis_title='Count',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
       
        if 'timestamp_utc' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = df_temp['timestamp_utc'].dt.date
            df_temp['hour'] = df_temp['timestamp_utc'].dt.hour
            
          
            heatmap_data = df_temp.groupby(['date', 'hour']).size().reset_index(name='count')
            pivot = heatmap_data.pivot(index='date', columns='hour', values='count').fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=[str(d) for d in pivot.index],
                colorscale='Viridis',
                colorbar=dict(title="Records")
            ))
            
            fig.update_layout(
                title='24-Hour Data Coverage Heatmap',
                xaxis_title='Hour of Day (UTC)',
                yaxis_title='Date',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
       
        if self.detected_timezones:
            st.info(f"🌐 **Detected Timezone Info:** {self.detected_timezones}")
        
        if self.dst_transitions:
            st.warning(f"⚠️ **DST Transitions Detected:** {len(self.dst_transitions)} potential transitions found")


class PreprocessingPipeline:
    """Main orchestrator for the complete preprocessing pipeline"""
    
    def __init__(self):
        self.loader = DataLoader()
        self.validator = DataValidator()
        self.aligner = TimeAligner()
        self.processed_data = {}
        self.reports = {}
        
    def run(self, uploaded_files, target_freq: str, fill_method: str) -> Dict[str, pd.DataFrame]:
        """Execute complete preprocessing pipeline"""
        
       
        st.header("📁 Data Loading")
        raw_data = self.loader.load_files(uploaded_files)
        
        if not raw_data:
            st.error("No valid data loaded. Please check your files.")
            return {}
        
      
        st.header("🔍 Data Validation & Cleaning")
        validated_data = {}
        
        for data_type, df in raw_data.items():
            with st.expander(f"📊 {data_type.replace('_', ' ').title()} Validation", expanded=False):
                cleaned_df, report = self.validator.validate(df, data_type)
                validated_data[data_type] = cleaned_df
                self.reports[f'{data_type}_validation'] = report
                self._display_validation_report(report)
        
      
        st.header("⏰ Time Alignment & Resampling")
        aligned_data = {}
        
        for data_type, df in validated_data.items():
            with st.expander(f"🔄 {data_type.replace('_', ' ').title()} Time Alignment", expanded=False):
                aligned_df, report = self.aligner.align(df, data_type, target_freq, fill_method)
                aligned_data[data_type] = aligned_df
                self.reports[f'{data_type}_alignment'] = report
                self._display_alignment_report(report)
        
        self.processed_data = aligned_data
        
   
        st.header("✅ Pipeline Complete")
        self._display_final_summary()
        
        return aligned_data
    
    def _display_validation_report(self, report: Dict):
        """Display validation results with metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Rows", f"{report['original_rows']:,}")
        with col2:
            st.metric("Final Rows", f"{report['final_rows']:,}")
        with col3:
            st.metric("Rows Removed", f"{report['rows_removed']:,}")
        with col4:
            quality_score = report['metrics'].get('quality_score', 0)
            st.metric("Quality Score", f"{quality_score}%", 
                     delta=f"{quality_score - 75:.1f}%" if quality_score >= 75 else None)
        
        if report['issues']:
            st.markdown("**⚠️ Issues Found:**")
            for issue in report['issues']:
                st.warning(issue)
        
        if report['metrics'].get('outliers_found'):
            st.markdown("**🎯 Outlier Detection:**")
            
            outlier_data = report['metrics']['outliers_found']
            
            # Build HTML table manually
            html_table = "<table style='width:100%; border-collapse: collapse;'>"
            html_table += "<tr style='background-color: #667eea; color: white;'><th>Metric</th><th>Count</th><th>Percentage</th><th>Bounds</th></tr>"
            
            for metric_name, metric_info in outlier_data.items():
                if isinstance(metric_info, dict):
                    count = metric_info.get('count', 'N/A')
                    percentage = metric_info.get('percentage', 'N/A')
                    bounds = metric_info.get('bounds', 'N/A')
                    
                    if isinstance(bounds, tuple):
                        bounds_str = f"({bounds[0]:.2f}, {bounds[1]:.2f})"
                    else:
                        bounds_str = str(bounds)
                    
                    html_table += f"<tr><td>{metric_name}</td><td>{count}</td><td>{percentage}</td><td>{bounds_str}</td></tr>"
            
            html_table += "</table>"
            st.markdown(html_table, unsafe_allow_html=True)

    
    def _display_alignment_report(self, report: Dict):
        """Display time alignment results"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Frequency", report['original_frequency'])
        with col2:
            st.metric("Target Frequency", report['target_frequency'])
        with col3:
            st.metric("Resampled Rows", f"{report.get('resampled_rows', 0):,}")
        
        if report.get('time_span'):
            st.info(f"⏱️ Time span: {report['time_span']}")
    
    def _display_final_summary(self):
        """Display comprehensive pipeline summary"""
        st.markdown("### 📊 Processing Summary")
        
        for data_type, df in self.processed_data.items():
            val_report = self.reports.get(f'{data_type}_validation', {})
            
            st.markdown(f"#### {data_type.replace('_', ' ').title()}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Records", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Quality", f"{val_report.get('metrics', {}).get('quality_score', 0)}%")
            with col4:
                time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                st.metric("Time Span", f"{time_span:.1f}h")
            with col5:
                st.metric("Status", "✅")
            
            st.markdown("---")

def load_sample_csv_files() -> List:
    """Load actual CSV files from sample_data folder"""
    import io
    import os

    sample_files = []
    sample_data_dir = "sample_data"

    # Map of expected files
    file_mappings = {
        'heartRateSample.csv': 'heart_rate_sample.csv',
        'sleep_sample.csv': 'sleep_sample.csv',
        'steps_sample.csv': 'steps_sample.csv'
    }

    for actual_file, display_name in file_mappings.items():
        file_path = os.path.join(sample_data_dir, actual_file)
        if os.path.exists(file_path):
            try:
                # Read the actual CSV file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Create a file-like object
                file_obj = io.StringIO(content)
                file_obj.name = display_name
                file_obj.seek(0)
                sample_files.append(file_obj)
                print(f"✅ Loaded {actual_file} as {display_name}")
            except Exception as e:
                print(f"❌ Error loading {actual_file}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

    return sample_files
    
    # Heart Rate Data (1-minute intervals, 12 hours)
    timestamps_hr = pd.date_range(start='2024-01-15 08:00:00', end='2024-01-15 20:00:00', freq='1min')
    hr_values = []
    
    for i, ts in enumerate(timestamps_hr):
        hour = ts.hour
        
        # Normal heart rate patterns throughout the day
        if 8 <= hour < 12:  # Morning
            base = 75
        elif 12 <= hour < 18:  # Afternoon
            base = 80
        else:  # Evening
            base = 70
        
        # Add natural variation
        hr = base + np.random.normal(0, 5) + 10 * np.sin(2 * np.pi * i / 120)
        hr_values.append(max(50, min(120, hr)))
    
    # Introduce anomalies
    hr_values[100:110] = [np.nan] * 10  # Missing data
    hr_values[300:310] = [130, 135, 140, 145, 150, 155, 150, 145, 140, 135]  # High heart rate spike (tachycardia)
    hr_values[500:505] = [35, 32, 28, 25, 30]  # Low heart rate (bradycardia)
    
    hr_df = pd.DataFrame({
        'timestamp': timestamps_hr,
        'heart_rate': hr_values
    })
    
    # Steps Data (15-minute intervals, 12 hours)
    timestamps_steps = pd.date_range(start='2024-01-15 08:00:00', end='2024-01-15 20:00:00', freq='15min')
    step_values = np.random.randint(50, 500, size=len(timestamps_steps))
    
    # Introduce step count anomalies
    step_values[10:12] = [1200, 1500]  # Unrealistic step counts
    
    steps_df = pd.DataFrame({
        'timestamp': timestamps_steps,
        'step_count': step_values
    })
    
    # Sleep Data (daily records for 7 days)
    sleep_dates = pd.date_range(start='2024-01-08', end='2024-01-14', freq='D')
    sleep_durations = []
    
    for i, date in enumerate(sleep_dates):
        # Normal sleep duration around 7-8 hours (420-480 minutes)
        base_sleep = np.random.normal(450, 60)  # Mean 450 min (7.5 hours), std 60 min
        
        # Introduce sleep anomalies
        if i == 2:  # Day 3: Too little sleep
            base_sleep = 120  # 2 hours - insomnia
        elif i == 5:  # Day 6: Too much sleep
            base_sleep = 800  # 13.3 hours - excessive sleep
        
        sleep_durations.append(max(0, base_sleep))
    
    sleep_df = pd.DataFrame({
        'date': sleep_dates,
        'duration_minutes': sleep_durations,
        'quality_score': np.random.uniform(0.6, 0.95, size=len(sleep_dates))
    })
    
    # Create file objects
    hr_file = io.StringIO()
    hr_df.to_csv(hr_file, index=False)
    hr_file.seek(0)
    hr_file.name = 'heart_rate_sample.csv'
    
    steps_file = io.StringIO()
    steps_df.to_csv(steps_file, index=False)
    steps_file.seek(0)
    steps_file.name = 'steps_sample.csv'
    
    sleep_file = io.StringIO()
    sleep_df.to_csv(sleep_file, index=False)
    sleep_file.seek(0)
    sleep_file.name = 'sleep_sample.csv'
    
    return [hr_file, steps_file, sleep_file]


class AdvancedAnalytics:
    """Advanced analytics and insights generation"""
    
    @staticmethod
    def detect_anomalies_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Detect anomalies using Z-score method"""
        if column not in df.columns:
            return df
        
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        df[f'{column}_anomaly_zscore'] = False
        df.loc[df[column].notna(), f'{column}_anomaly_zscore'] = z_scores > threshold
        return df
    
    @staticmethod
    def detect_peaks_valleys(df: pd.DataFrame, column: str) -> Dict:
        """Detect peaks and valleys in time series data"""
        if column not in df.columns:
            return {}
        
        data = df[column].dropna().values
        peaks, peak_props = find_peaks(data, prominence=1)
        valleys, valley_props = find_peaks(-data, prominence=1)
        
        return {
            'peaks': peaks,
            'valleys': valleys,
            'peak_count': len(peaks),
            'valley_count': len(valleys),
            'avg_peak_value': data[peaks].mean() if len(peaks) > 0 else 0,
            'avg_valley_value': data[valleys].mean() if len(valleys) > 0 else 0
        }
    
    @staticmethod
    def calculate_trends(df: pd.DataFrame, column: str) -> Dict:
        """Calculate trend statistics"""
        if column not in df.columns or len(df) < 2:
            return {}
        
        values = df[column].dropna().values
        x = np.arange(len(values))
        
        if len(values) < 2:
            return {}
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        return {
            'slope': slope,
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
    @staticmethod
    def generate_health_insights(df: pd.DataFrame, data_type: str) -> List[str]:
        """Generate AI-powered health insights"""
        insights = []
        
        if data_type == 'heart_rate' and 'heart_rate' in df.columns:
            avg_hr = df['heart_rate'].mean()
            max_hr = df['heart_rate'].max()
            min_hr = df['heart_rate'].min()
            std_hr = df['heart_rate'].std()
            
            if avg_hr < 60:
                insights.append(f"💙 Your average heart rate ({avg_hr:.0f} bpm) is below normal. This could indicate good cardiovascular fitness or bradycardia.")
            elif avg_hr > 100:
                insights.append(f"❤️ Your average heart rate ({avg_hr:.0f} bpm) is elevated. Consider consulting a healthcare provider.")
            else:
                insights.append(f"✅ Your average heart rate ({avg_hr:.0f} bpm) is within normal range.")
            
            if std_hr > 15:
                insights.append(f"📊 High heart rate variability detected (σ={std_hr:.1f}). This suggests good adaptability to stress.")
            
            if max_hr > 180:
                insights.append(f"⚠️ Peak heart rate of {max_hr:.0f} bpm detected. Ensure this occurred during intense exercise.")
        
        elif data_type == 'steps' and 'step_count' in df.columns:
            total_steps = df['step_count'].sum()
            avg_steps = df['step_count'].mean()
            daily_goal = 10000
            
            if total_steps >= daily_goal:
                insights.append(f"🎯 Excellent! You've achieved {total_steps:,.0f} steps, exceeding the daily goal of {daily_goal:,}!")
            else:
                remaining = daily_goal - total_steps
                insights.append(f"🚶 You have {total_steps:,.0f} steps. {remaining:,.0f} more to reach your daily goal!")
            
            if avg_steps > 500:
                insights.append(f"💪 Great activity level! Averaging {avg_steps:.0f} steps per interval.")
        
        elif data_type == 'sleep' and 'sleep_stage' in df.columns:
            stage_counts = df['sleep_stage'].value_counts()
            total_duration = df['duration_minutes'].sum() if 'duration_minutes' in df.columns else 0
            
            if total_duration > 0:
                hours = total_duration / 60
                if hours >= 7 and hours <= 9:
                    insights.append(f"😴 Optimal sleep duration: {hours:.1f} hours. Well done!")
                elif hours < 7:
                    insights.append(f"⏰ Sleep duration ({hours:.1f}h) is below recommended 7-9 hours.")
                else:
                    insights.append(f"😴 Long sleep duration ({hours:.1f}h). Quality matters more than quantity!")
            
            if 'deep' in stage_counts:
                deep_pct = (stage_counts.get('deep', 0) / len(df)) * 100
                if deep_pct >= 20:
                    insights.append(f"🌙 Excellent deep sleep: {deep_pct:.0f}% of total sleep time.")
        
        return insights


class DataComparison:
    """Compare multiple datasets and generate comparative insights"""
    
    @staticmethod
    def compare_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate comparison statistics across datasets"""
        comparison_data = []
        
        for name, df in datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if not col.endswith('_is_outlier') and not col.endswith('_anomaly'):
                    comparison_data.append({
                        'Dataset': name.replace('_', ' ').title(),
                        'Metric': col.replace('_', ' ').title(),
                        'Mean': df[col].mean(),
                        'Median': df[col].median(),
                        'Std Dev': df[col].std(),
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Records': len(df)
                    })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df[[col for col in numeric_df.columns 
                                if not col.endswith('_is_outlier') and not col.endswith('_anomaly')]]
        
        if numeric_df.shape[1] < 2:
            return None
        
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        
        return fig


class DataQualityReport:
    """Generate comprehensive data quality reports"""
    
    @staticmethod
    def generate_report(df: pd.DataFrame, data_type: str) -> Dict:
        """Generate detailed quality report"""
        report = {
            'data_type': data_type,
            'total_records': len(df),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'completeness': {},
            'uniqueness': {},
            'data_types': {}
        }
        
     
        for col in df.columns:
            non_null = df[col].notna().sum()
            report['completeness'][col] = {
                'non_null': non_null,
                'null': len(df) - non_null,
                'completeness_pct': (non_null / len(df)) * 100
            }
        
      
        for col in df.columns:
            unique_count = df[col].nunique()
            report['uniqueness'][col] = {
                'unique_values': unique_count,
                'uniqueness_pct': (unique_count / len(df)) * 100
            }
        
      
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
        
        return report
    
    @staticmethod
    def create_quality_dashboard(report: Dict) -> None:
        """Display quality dashboard"""
        st.markdown("### 📊 Data Quality Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{report['total_records']:,}")
        with col2:
            st.metric("Columns", len(report['columns']))
        with col3:
            st.metric("Memory Usage", f"{report['memory_usage_mb']:.2f} MB")
        with col4:
            avg_completeness = np.mean([v['completeness_pct'] for v in report['completeness'].values()])
            st.metric("Avg Completeness", f"{avg_completeness:.1f}%")
        
      
        completeness_df = pd.DataFrame([
            {'Column': k, 'Completeness %': v['completeness_pct']}
            for k, v in report['completeness'].items()
        ])
        
        fig = px.bar(completeness_df, x='Column', y='Completeness %',
                    title='Data Completeness by Column',
                    color='Completeness %',
                    color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def create_advanced_visualization(df: pd.DataFrame, data_type: str):
    """Create advanced interactive visualizations with AI insights"""
    
  
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                    padding: 1.5rem; border-radius: 16px; border-left: 5px solid #667eea; margin-bottom: 1.5rem;">
            <h3 style="margin: 0; color: #495057; font-size: 1.5rem; font-weight: 700;">
                📈 {data_type.replace('_', ' ').title()} - Advanced Analytics Dashboard
            </h3>
            <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.95rem;">
                Interactive visualizations with AI-powered insights and anomaly detection
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if not col.endswith('_is_outlier') and not col.endswith('_anomaly')]
    
    if not numeric_cols:
        st.warning("No numeric data to visualize")
        return
    
    
    col_select, viz_type, show_stats = st.columns([2, 1, 1])
    with col_select:
        selected_col = st.selectbox(f"📊 Select Metric:", numeric_cols, key=f"viz_{data_type}")
    with viz_type:
        chart_type = st.selectbox("📈 Chart Type:", ["Time Series", "Distribution", "Box Plot", "Violin Plot"], key=f"chart_{data_type}")
    with show_stats:
        show_advanced = st.checkbox("🔍 Show Advanced Stats", value=True, key=f"stats_{data_type}")
    
   
    df_analyzed = AdvancedAnalytics.detect_anomalies_zscore(df.copy(), selected_col)
    peaks_valleys = AdvancedAnalytics.detect_peaks_valleys(df_analyzed, selected_col)
    trends = AdvancedAnalytics.calculate_trends(df_analyzed, selected_col)
    
   
    if show_advanced:
        st.markdown("#### 🎯 Quick Statistics Preview")
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        with quick_col1:
            st.metric("📊 Data Points", f"{len(df_analyzed):,}")
        with quick_col2:
            anomaly_count = df_analyzed[f'{selected_col}_anomaly_zscore'].sum() if f'{selected_col}_anomaly_zscore' in df_analyzed.columns else 0
            st.metric("⚠️ Anomalies", f"{int(anomaly_count)}", delta=f"{(anomaly_count/len(df_analyzed)*100):.1f}%")
        with quick_col3:
            st.metric("🔺 Peaks", f"{peaks_valleys.get('peak_count', 0)}")
        with quick_col4:
            st.metric("🔻 Valleys", f"{peaks_valleys.get('valley_count', 0)}")
        st.markdown("---")
    
  
    if chart_type == "Time Series":
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{selected_col.replace("_", " ").title()} Over Time with Anomalies', 'Distribution Analysis'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
       
        fig.add_trace(
            go.Scatter(x=df_analyzed['timestamp'], y=df_analyzed[selected_col], 
                      mode='lines', name='Values',
                      line=dict(color='#667eea', width=2.5),
                      fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'),
            row=1, col=1
        )
        
       
        anomaly_col = f'{selected_col}_anomaly_zscore'
        if anomaly_col in df_analyzed.columns:
            anomaly_data = df_analyzed[df_analyzed[anomaly_col] == True]
            if not anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(x=anomaly_data['timestamp'], y=anomaly_data[selected_col],
                              mode='markers', name='Z-Score Anomalies',
                              marker=dict(color='#f5576c', size=10, symbol='diamond',
                                        line=dict(color='white', width=2))),
                    row=1, col=1
                )
        
       
        if peaks_valleys.get('peak_count', 0) > 0:
            peak_indices = peaks_valleys['peaks']
            fig.add_trace(
                go.Scatter(x=df_analyzed.iloc[peak_indices]['timestamp'], 
                          y=df_analyzed.iloc[peak_indices][selected_col],
                          mode='markers', name='Peaks',
                          marker=dict(color='#28a745', size=8, symbol='triangle-up')),
                row=1, col=1
            )
        
        if peaks_valleys.get('valley_count', 0) > 0:
            valley_indices = peaks_valleys['valleys']
            fig.add_trace(
                go.Scatter(x=df_analyzed.iloc[valley_indices]['timestamp'], 
                          y=df_analyzed.iloc[valley_indices][selected_col],
                          mode='markers', name='Valleys',
                          marker=dict(color='#17a2b8', size=8, symbol='triangle-down')),
                row=1, col=1
            )
        
      
        fig.add_trace(
            go.Histogram(x=df_analyzed[selected_col], name='Distribution',
                        marker=dict(color='#764ba2'), nbinsx=30, showlegend=False),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        fig.update_xaxes(title_text=selected_col.replace('_', ' ').title(), row=2, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(title_text="Value", row=1, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(title_text="Frequency", row=2, col=1, showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        fig.update_layout(
            height=850,
            showlegend=True,
            hovermode='x unified',
            font=dict(family='Inter, sans-serif', size=12),
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#667eea',
                borderwidth=2,
                font=dict(size=12)
            )
        )
        
    elif chart_type == "Distribution":
        fig = go.Figure()
        clean_data = df_analyzed[selected_col].dropna()
        
        if len(clean_data) == 0:
            st.warning("No valid data available for distribution plot")
            return
        
        fig.add_trace(go.Histogram(
            x=clean_data, 
            nbinsx=50,
            marker=dict(
                color='#667eea', 
                line=dict(color='white', width=1)
            ),
            name='Frequency'
        ))
      
        mean_val = clean_data.mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#f5576c", 
                     annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
       
        median_val = clean_data.median()
        fig.add_vline(x=median_val, line_dash="dash", line_color="#28a745", 
                     annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom")
        
        fig.update_layout(
            title=f'📊 {selected_col.replace("_", " ").title()} Distribution with Mean & Median',
            xaxis_title=selected_col.replace('_', ' ').title(),
            yaxis_title='Frequency', 
            height=600,
            showlegend=False,
            font=dict(family='Inter, sans-serif', size=13),
            title_font=dict(size=18, color='#1e3a8a'),
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        )
    
    elif chart_type == "Box Plot":
        fig = go.Figure()
       
        clean_data = df_analyzed[selected_col].dropna()
        
        if len(clean_data) == 0:
            st.warning("No valid data available for box plot")
            return
        
        fig.add_trace(go.Box(
            y=clean_data, 
            name=selected_col.replace('_', ' ').title(),
            marker=dict(color='#764ba2'),
            boxmean='sd',
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
        fig.update_layout(
            title=f'📦 {selected_col.replace("_", " ").title()} Box Plot with Outliers',
            yaxis_title=selected_col.replace('_', ' ').title(), 
            height=600,
            showlegend=False,
            font=dict(family='Inter, sans-serif', size=13),
            title_font=dict(size=18, color='#1e3a8a'),
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        )
    
    else:  
        fig = go.Figure()
        clean_data = df_analyzed[selected_col].dropna()
        
        if len(clean_data) == 0:
            st.warning("No valid data available for violin plot")
            return
        
        fig.add_trace(go.Violin(
            y=clean_data, 
            name=selected_col.replace('_', ' ').title(),
            box_visible=True, 
            meanline_visible=True,
            fillcolor='#f093fb', 
            line_color='#764ba2',
            points='outliers',
            jitter=0.3,
            pointpos=-0.5
        ))
        fig.update_layout(
            title=f'🎻 {selected_col.replace("_", " ").title()} Violin Plot with Distribution',
            yaxis_title=selected_col.replace('_', ' ').title(), 
            height=600,
            showlegend=False,
            font=dict(family='Inter, sans-serif', size=13),
            title_font=dict(size=18, color='#1e3a8a'),
            plot_bgcolor='rgba(255,255,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        )
    
    st.plotly_chart(fig, use_container_width=True)
  
    st.markdown("### 📊 Statistical Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mean", f"{df_analyzed[selected_col].mean():.2f}")
    with col2:
        st.metric("Median", f"{df_analyzed[selected_col].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df_analyzed[selected_col].std():.2f}")
    with col4:
        st.metric("Range", f"{df_analyzed[selected_col].max() - df_analyzed[selected_col].min():.2f}")
    with col5:
        if trends.get('is_significant'):
            trend_emoji = "📈" if trends['trend'] == 'increasing' else "📉"
            st.metric("Trend", f"{trend_emoji} {trends['trend'].title()}")
        else:
            st.metric("Trend", "➡️ Stable")
    
   
    st.markdown("### 🔍 Advanced Analytics Results")
    
   
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        anomaly_count = df_analyzed[f'{selected_col}_anomaly_zscore'].sum() if f'{selected_col}_anomaly_zscore' in df_analyzed.columns else 0
        anomaly_pct = (anomaly_count / len(df_analyzed) * 100) if len(df_analyzed) > 0 else 0
        st.metric("⚠️ Z-Score Anomalies", f"{int(anomaly_count)}", delta=f"{anomaly_pct:.1f}% of data")
    
    with col2:
        peak_count = peaks_valleys.get('peak_count', 0)
        avg_peak = peaks_valleys.get('avg_peak_value', 0)
        st.metric("🔺 Peaks Detected", f"{peak_count}", delta=f"Avg: {avg_peak:.1f}" if peak_count > 0 else None)
    
    with col3:
        valley_count = peaks_valleys.get('valley_count', 0)
        avg_valley = peaks_valleys.get('avg_valley_value', 0)
        st.metric("🔻 Valleys Detected", f"{valley_count}", delta=f"Avg: {avg_valley:.1f}" if valley_count > 0 else None)
    
    with col4:
        if trends.get('is_significant'):
            r_squared = trends.get('r_squared', 0)
            st.metric("📈 Trend Strength", f"{r_squared:.3f}", delta="Significant" if r_squared > 0.5 else "Weak")
        else:
            st.metric("📈 Trend Strength", "N/A", delta="Not significant")
    
    if show_advanced and trends.get('is_significant'):
        st.markdown("#### 📊 Trend Analysis Details")
        trend_col1, trend_col2, trend_col3 = st.columns(3)
        with trend_col1:
            st.info(f"**Slope:** {trends.get('slope', 0):.4f}")
        with trend_col2:
            st.info(f"**R² Value:** {trends.get('r_squared', 0):.4f}")
        with trend_col3:
            st.info(f"**P-Value:** {trends.get('p_value', 0):.4e}")
    
  
    insights = AdvancedAnalytics.generate_health_insights(df_analyzed, data_type)
    if insights:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(0, 242, 254, 0.1)); 
                        padding: 1rem; border-radius: 12px; border-left: 4px solid #4facfe; margin: 1.5rem 0;">
                <h4 style="margin: 0 0 0.5rem 0; color: #495057;">🤖 AI-Powered Health Insights</h4>
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                    Personalized recommendations based on your data patterns
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        for idx, insight in enumerate(insights):
            st.info(insight)
    
    
    quality_report = DataQualityReport.generate_report(df_analyzed, data_type)
    with st.expander("📋 View Detailed Quality Report"):
        DataQualityReport.create_quality_dashboard(quality_report)


def create_visualization(df: pd.DataFrame, data_type: str):
    """Wrapper to use advanced visualization"""
    #create_advanced_visualization(df, data_type)


def export_processed_data(processed_data: Dict[str, pd.DataFrame]):
    """Export processed data to CSV with premium styling"""
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0 1rem 0;">
           <h2 style="font-size: 2rem; font-weight: 800; color: black;">
               💾 Export Processed Data
           </h2>
            <p style="color: #6c757d; font-size: 1rem; margin-top: 0.5rem;">
                Download your cleaned and processed datasets
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(processed_data))
    for idx, (data_type, df) in enumerate(processed_data.items()):
        with cols[idx]:
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {data_type.replace('_', ' ').title()}",
                data=csv,
                file_name=f"fitpulse_{data_type}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{data_type}",
                use_container_width=True
            )
            st.caption(f"📊 {len(df):,} records")


# ============================================================================
# MILESTONE 2 UI INTEGRATION
# ============================================================================

class TSFreshFeatureExtractor:
    """Extract time-series features using TSFresh library"""
    
    def __init__(self, feature_complexity: str = 'minimal'):
        """
        Initialize feature extractor
        
        Args:
            feature_complexity: 'minimal', 'efficient', or 'comprehensive'
        """
        self.feature_complexity = feature_complexity
        self.feature_matrix = None
        self.feature_names = []
        self.extraction_report = {}
        
    def extract_features(self, df: pd.DataFrame, data_type: str, 
                        window_size: int = 60) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract statistical features from time-series data
        
        Args:
            df: Preprocessed dataframe with timestamp and metric columns
            data_type: Type of data ('heart_rate', 'steps', 'sleep')
            window_size: Rolling window size in minutes for feature extraction
            
        Returns:
            Feature matrix dataframe and extraction report
        """
        st.info(f"🔄 Extracting features from {data_type} data...")
        
        # Initialize actual_window_size to requested window_size
        actual_window_size = window_size
        
        report = {
            'data_type': data_type,
            'original_rows': len(df),
            'window_size': actual_window_size,  # Will be updated if adaptive sizing is used
            'requested_window_size': window_size,
            'features_extracted': 0,
            'extraction_time': None,
            'success': False
        }
        
        try:
            # Prepare data for TSFresh
            df_prepared, actual_window_size = self._prepare_data_for_tsfresh(df, data_type, window_size)
            
            # Update report with actual window size
            report['window_size'] = actual_window_size
            
            if df_prepared is None or len(df_prepared) == 0:
                report['error'] = "No data available for feature extraction"
                return pd.DataFrame(), report
            
            # Select feature extraction parameters
            if self.feature_complexity == 'minimal':
                from tsfresh.feature_extraction import MinimalFCParameters
                fc_parameters = MinimalFCParameters()
            elif self.feature_complexity == 'comprehensive':
                from tsfresh.feature_extraction import ComprehensiveFCParameters
                fc_parameters = ComprehensiveFCParameters()
            else:  # efficient - custom selection
                fc_parameters = self._get_efficient_parameters()
            
            start_time = datetime.now()
            
            # Extract features (suppress progress bars for better performance)
            import logging
            logging.getLogger('tsfresh').setLevel(logging.WARNING)  # Suppress TSFresh info logs

            from tsfresh import extract_features
            feature_matrix = extract_features(
                df_prepared,
                column_id='window_id',
                column_sort='timestamp',
                default_fc_parameters=fc_parameters,
                disable_progressbar=True,
                n_jobs=1,
                show_warnings=False  # Suppress warnings for cleaner output
            )
            
            # Handle missing values
            from tsfresh.utilities.dataframe_functions import impute
            feature_matrix = impute(feature_matrix)
            
            # Remove constant features
            feature_matrix = self._remove_constant_features(feature_matrix)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            self.feature_matrix = feature_matrix
            self.feature_names = list(feature_matrix.columns)
            
            report['features_extracted'] = len(self.feature_names)
            report['extraction_time'] = extraction_time
            report['feature_windows'] = len(feature_matrix)
            report['success'] = True
            
            self.extraction_report = report
            
          
            
            return feature_matrix, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Feature extraction failed: {str(e)}")
            return pd.DataFrame(), report
    
    def _prepare_data_for_tsfresh(self, df: pd.DataFrame, data_type: str,
                                  window_size: int) -> Tuple[pd.DataFrame, int]:
        """Prepare data in TSFresh format with rolling windows and return actual window size used"""
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Special handling for sleep data - it's daily aggregated, not time-series
        if data_type == 'sleep':
            st.info("💤 Sleep data is daily aggregated. Using daily statistics instead of time-series features.")
            # For sleep data, aggregate back to daily since it may have been resampled to high frequency
            if len(df) >= 3:  # Need at least 3 days for meaningful stats
                # Resample to daily to get daily sleep duration
                df_daily = df.set_index('timestamp').resample('D')['duration_minutes'].sum().reset_index()
                
                if len(df_daily) >= 3:
                    df_prepared = pd.DataFrame({
                        'window_id': [0] * len(df_daily),
                        'timestamp': df_daily['timestamp'],
                        'value': df_daily['duration_minutes']
                    })
                    return df_prepared, window_size
                else:
                    st.warning("⚠️ Sleep data has too few daily records for analysis (need at least 3 days)")
                    return None, window_size
            else:
                st.warning("⚠️ Sleep data has too few records for analysis (need at least 3 days)")
                return None, window_size

        # Identify the metric column
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }

        if data_type not in metric_columns:
            return None, window_size

        metric_col = metric_columns[data_type]

        if metric_col not in df.columns:
            st.warning(f"Metric column '{metric_col}' not found in dataframe")
            return None, window_size

        # Create rolling windows with adaptive sizing
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Adaptive window sizing: if not enough data for requested window, use smaller windows
        min_window_size = 10  # Minimum 10 records for meaningful features
        actual_window_size = window_size
        
        # Reduce window size if not enough data
        while actual_window_size > min_window_size and len(df_sorted) < actual_window_size:
            actual_window_size = actual_window_size // 2
        
        if len(df_sorted) < min_window_size:
            st.warning(f"⚠️ Not enough data for feature extraction. Need at least {min_window_size} records, got {len(df_sorted)}.")
            return None, window_size
        
        if actual_window_size != window_size:
            st.info(f"📊 Using smaller window size ({actual_window_size} min) for {data_type} due to limited data")

        prepared_data = []
        window_id = 0

        # Create overlapping windows (50% overlap)
        step_size = max(1, actual_window_size // 2)

        for i in range(0, len(df_sorted) - actual_window_size + 1, step_size):
            window_data = df_sorted.iloc[i:i+actual_window_size].copy()
            window_data['window_id'] = window_id
            prepared_data.append(window_data[['window_id', 'timestamp', metric_col]])
            window_id += 1

        if not prepared_data:
            st.warning(f"⚠️ Could not create any windows for {data_type} data")
            return None, actual_window_size
        
        df_prepared = pd.concat(prepared_data, ignore_index=True)
        df_prepared = df_prepared.rename(columns={metric_col: 'value'})
        
        return df_prepared, actual_window_size
    
    def _get_efficient_parameters(self):
        """Get efficient feature set - balance between minimal and comprehensive"""
        return {
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "minimum": None,
            "maximum": None,
            "quantile": [{"q": 0.25}, {"q": 0.75}],
            "skewness": None,
            "kurtosis": None,
            "abs_energy": None,
            "absolute_sum_of_changes": None,
            "count_above_mean": None,
            "count_below_mean": None,
            "linear_trend": [{"attr": "slope"}],
            "autocorrelation": [{"lag": 1}, {"lag": 2}],
            "c3": [{"lag": 1}],
            "approximate_entropy": [{"m": 2, "r": 0.2}]
        }
    
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance"""
        constant_features = [col for col in df.columns if df[col].std() == 0]
        if constant_features:
            st.info(f"Removed {len(constant_features)} constant features")
            df = df.drop(columns=constant_features)
        return df
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N most variable features"""
        if self.feature_matrix is None or self.feature_matrix.empty:
            return pd.DataFrame()
        
        # Calculate variance for each feature
        feature_variance = self.feature_matrix.var().sort_values(ascending=False)
        top_features = feature_variance.head(n)
        
        return pd.DataFrame({
            'Feature': top_features.index,
            'Variance': top_features.values,
            'Mean': [self.feature_matrix[feat].mean() for feat in top_features.index],
            'Std': [self.feature_matrix[feat].std() for feat in top_features.index]
        })
    
    def visualize_feature_distributions(self, n_features: int = 10):
        """Visualize top N feature distributions"""
        if self.feature_matrix is None or self.feature_matrix.empty:
            st.warning("No features extracted yet")
            return
        
        top_features_df = self.get_top_features(n_features)
        
        if top_features_df.empty:
            return
        
        # Create subplots
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=top_features_df['Feature'].tolist()
        )
        
        for idx, feature in enumerate(top_features_df['Feature']):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.feature_matrix[feature],
                    name=feature,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Top Feature Distributions",
            height=300 * n_rows,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


class ProphetTrendModeler:
    """Model time-series trends using Facebook Prophet"""
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.residuals = {}
        self.modeling_reports = {}
    def fit_and_predict(self, df: pd.DataFrame, data_type: str,
                       forecast_periods: int = 100) -> Tuple[pd.DataFrame, Dict]:
        st.info(f"🔄 Modeling trends for {data_type} data with Prophet...")
        report = {
            'data_type': data_type,
            'training_rows': len(df),
            'forecast_periods': forecast_periods,
            'success': False
        }
        try:
            metric_columns = {
                'heart_rate': 'heart_rate',
                'steps': 'step_count',
                'sleep': 'duration_minutes'
            }
            if data_type not in metric_columns:
                report['error'] = f"Unknown data type: {data_type}"
                return pd.DataFrame(), report
            metric_col = metric_columns[data_type]
            if metric_col not in df.columns:
                report['error'] = f"Metric column '{metric_col}' not found"
                return pd.DataFrame(), report
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['timestamp']).dt.tz_localize(None) if pd.api.types.is_datetime64_any_dtype(df['timestamp']) else df['timestamp'],
                'y': df[metric_col]
            })
            prophet_df = prophet_df.dropna()
            if len(prophet_df) < 2:
                report['error'] = "Insufficient data for modeling"
                return pd.DataFrame(), report
            from prophet import Prophet
            import prophet
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            st.write("Fitting Prophet model...")
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_periods, freq='min')
            forecast = model.predict(future)
            merged = prophet_df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                                     on='ds', how='left')
            merged['residual'] = merged['y'] - merged['yhat']
            merged['residual_abs'] = np.abs(merged['residual'])
            self.models[data_type] = model
            self.forecasts[data_type] = forecast
            self.residuals[data_type] = merged
            report['mae'] = merged['residual_abs'].mean()
            report['rmse'] = np.sqrt((merged['residual'] ** 2).mean())
            report['mape'] = (merged['residual_abs'] / merged['y'].abs()).mean() * 100
            report['success'] = True
            self.modeling_reports[data_type] = report
            st.success(f"✅ Prophet model trained successfully")
            st.write(f"MAE: {report['mae']:.2f}, RMSE: {report['rmse']:.2f}, MAPE: {report['mape']:.2f}%")
            return forecast, report
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Prophet modeling failed: {str(e)}")
            return pd.DataFrame(), report
    def visualize_forecast(self, data_type: str, df_original: pd.DataFrame):
        if data_type not in self.forecasts:
            st.warning(f"No forecast available for {data_type}")
            return
        forecast = self.forecasts[data_type]
        residuals = self.residuals[data_type]
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        metric_col = metric_columns.get(data_type)
        if metric_col not in df_original.columns:
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_original['timestamp'],
            y=df_original[metric_col],
            mode='markers',
            name='Actual',
            marker=dict(size=4, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(width=0),
            showlegend=False
        ))
        fig.update_layout(
            title=f"Prophet Forecast: {data_type.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=metric_col.replace('_', ' ').title(),
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        self._visualize_residuals(data_type)
    def _visualize_residuals(self, data_type: str):
        if data_type not in self.residuals:
            return
        residuals = self.residuals[data_type]
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Residuals Over Time', 'Residual Distribution'),
            row_heights=[0.6, 0.4]
        )
        fig.add_trace(
            go.Scatter(
                x=residuals['ds'],
                y=residuals['residual'],
                mode='markers',
                name='Residuals',
                marker=dict(size=4, color='purple')
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_trace(
            go.Histogram(
                x=residuals['residual'],
                name='Distribution',
                nbinsx=50,
                marker=dict(color='purple')
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_xaxes(title_text="Residual Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_layout(
            title_text="Prophet Model Residuals (for Anomaly Detection)",
            height=700,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    def get_anomalies_from_residuals(self, data_type: str, 
                                    threshold_std: float = 3.0) -> pd.DataFrame:
        if data_type not in self.residuals:
            return pd.DataFrame()
        residuals = self.residuals[data_type]
        mean_residual = residuals['residual'].mean()
        std_residual = residuals['residual'].std()
        threshold = threshold_std * std_residual
        anomalies = residuals[
            (residuals['residual'] > mean_residual + threshold) |
            (residuals['residual'] < mean_residual - threshold)
        ].copy()
        anomalies['anomaly_score'] = np.abs(anomalies['residual'] - mean_residual) / std_residual
        return anomalies



class BehaviorClusterer:
   
    
    def __init__(self):
        self.scalers = {}
        self.kmeans_models = {}
        self.dbscan_models = {}
        self.cluster_labels = {}
        self.clustering_reports = {}
        self.reduced_features = {}
    
    def cluster_features(self, feature_matrix: pd.DataFrame, data_type: str,
                        method: str = 'kmeans', n_clusters: int = 3,
                        eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Cluster feature vectors to identify behavioral patterns
        
        Args:
            feature_matrix: TSFresh feature matrix
            data_type: Type of data
            method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters for KMeans
            eps: Epsilon parameter for DBSCAN
            min_samples: Min samples for DBSCAN
            
        Returns:
            Cluster labels and clustering report
        """
        st.info(f"🔄 Clustering {data_type} patterns using {method.upper()}...")
        
        report = {
            'data_type': data_type,
            'method': method,
            'n_samples': len(feature_matrix),
            'n_features': len(feature_matrix.columns),
            'success': False
        }
        
        try:
            if feature_matrix.empty:
                report['error'] = "Empty feature matrix"
                return np.array([]), report
            
            n_samples = len(feature_matrix)
            
            # Check if we have enough samples for meaningful clustering
            if n_samples < 2:
                report['error'] = f"Insufficient data: only {n_samples} sample(s) available for clustering. Need at least 2 samples."
                st.warning(f"⚠️ {data_type}: Insufficient data for clustering analysis (only {n_samples} sample). Skipping clustering.")
                return np.array([]), report
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            self.scalers[data_type] = scaler
            
            # Apply clustering
            if method == 'kmeans':
                labels, model_report = self._apply_kmeans(
                    features_scaled, n_clusters, data_type
                )
                report.update(model_report)
                
            elif method == 'dbscan':
                labels, model_report = self._apply_dbscan(
                    features_scaled, eps, min_samples, data_type
                )
                report.update(model_report)
            else:
                report['error'] = f"Unknown clustering method: {method}"
                return np.array([]), report
            
            self.cluster_labels[data_type] = labels
            
            # Calculate clustering metrics
            if len(np.unique(labels)) > 1:
                try:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score
                    # davies_bouldin_score requires: 2 <= n_labels <= n_samples - 1
                    n_labels = len(np.unique(labels))
                    n_samples = len(features_scaled)
                    
                    if n_labels < 2 or n_labels > n_samples - 1:
                        # Skip davies_bouldin_score if constraints aren't met
                        silhouette = silhouette_score(features_scaled, labels)
                        report['silhouette_score'] = silhouette
                    else:
                        silhouette = silhouette_score(features_scaled, labels)
                        davies_bouldin = davies_bouldin_score(features_scaled, labels)
                        report['silhouette_score'] = silhouette
                        report['davies_bouldin_score'] = davies_bouldin
                except Exception as metric_error:
                    # Skip metrics if calculation fails
                    pass

            report['n_clusters'] = len(np.unique(labels))
            report['cluster_sizes'] = {
                int(label): int(count) 
                for label, count in zip(*np.unique(labels, return_counts=True))
            }
            report['success'] = True
            
            self.clustering_reports[data_type] = report
            
            st.success(f"✅ Identified {report['n_clusters']} clusters")
            
            return labels, report
            
        except Exception as e:
            report['error'] = str(e)
            st.error(f"❌ Clustering failed: {str(e)}")
            return np.array([]), report
    
    def _apply_kmeans(self, features_scaled: np.ndarray, n_clusters: int,
                     data_type: str) -> Tuple[np.ndarray, Dict]:
        """Apply KMeans clustering"""
        
        from sklearn.cluster import KMeans
        
        # Adjust n_clusters if we have fewer samples than requested clusters
        n_samples = len(features_scaled)
        if n_samples < n_clusters:
            if n_samples == 1:
                # For single sample, create one cluster
                actual_clusters = 1
                st.warning(f"⚠️ {data_type}: Only {n_samples} sample available. Using 1 cluster instead of {n_clusters}.")
            else:
                # Use fewer clusters than requested
                actual_clusters = n_samples
                st.warning(f"⚠️ {data_type}: Only {n_samples} samples available. Using {actual_clusters} clusters instead of {n_clusters}.")
        else:
            actual_clusters = n_clusters
        
        model = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features_scaled)
        
        self.kmeans_models[data_type] = model
        
        report = {
            'inertia': model.inertia_,
            'n_iterations': model.n_iter_,
            'requested_clusters': n_clusters,
            'actual_clusters': actual_clusters
        }
        
        return labels, report
    
    def _apply_dbscan(self, features_scaled: np.ndarray, eps: float,
                     min_samples: int, data_type: str) -> Tuple[np.ndarray, Dict]:
        """Apply DBSCAN clustering"""
        
        from sklearn.cluster import DBSCAN
        
        n_samples = len(features_scaled)
        
        # Adjust min_samples if we have very few samples
        if n_samples < min_samples:
            actual_min_samples = max(1, n_samples // 2)  # Use at most half the samples
            st.warning(f"⚠️ {data_type}: Only {n_samples} samples available. Using min_samples={actual_min_samples} instead of {min_samples}.")
        else:
            actual_min_samples = min_samples
        
        model = DBSCAN(eps=eps, min_samples=actual_min_samples)
        labels = model.fit_predict(features_scaled)
        
        self.dbscan_models[data_type] = model
        
        n_noise = np.sum(labels == -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        report = {
            'eps': eps,
            'min_samples': min_samples,
            'actual_min_samples': actual_min_samples,
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'noise_percentage': (n_noise / len(labels)) * 100
        }
        
        return labels, report
    
    def reduce_and_visualize(self, feature_matrix: pd.DataFrame, data_type: str,
                           method: str = 'pca', n_components: int = 2):
        """
        Reduce dimensions and visualize clusters
        
        Args:
            feature_matrix: Feature matrix
            data_type: Type of data
            method: 'pca' or 'tsne'
            n_components: Number of dimensions
        """
        if data_type not in self.cluster_labels:
            st.warning("No clustering results available. Run clustering first.")
            return
        
        if feature_matrix.empty:
            st.warning("Empty feature matrix")
            return
        
        labels = self.cluster_labels[data_type]
        
        # Get scaled features
        if data_type in self.scalers:
            features_scaled = self.scalers[data_type].transform(feature_matrix)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
        
        # Dimension reduction
        st.info(f"Reducing dimensions using {method.upper()}...")
        
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
            features_reduced = reducer.fit_transform(features_scaled)
            
            explained_var = reducer.explained_variance_ratio_
            st.write(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
            
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
            features_reduced = reducer.fit_transform(features_scaled)
        else:
            st.error(f"Unknown method: {method}")
            return
        
        self.reduced_features[data_type] = features_reduced
        
        # Visualize
        self._visualize_clusters_2d(features_reduced, labels, data_type, method)
    
    def _visualize_clusters_2d(self, features_reduced: np.ndarray, 
                              labels: np.ndarray, data_type: str, method: str):
        """Visualize clusters in 2D"""
        
        df_viz = pd.DataFrame({
            'Component 1': features_reduced[:, 0],
            'Component 2': features_reduced[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # Create scatter plot
        fig = px.scatter(
            df_viz,
            x='Component 1',
            y='Component 2',
            color='Cluster',
            title=f"Cluster Visualization: {data_type.replace('_', ' ').title()} ({method.upper()})",
            labels={'Cluster': 'Cluster ID'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster statistics
        self._show_cluster_statistics(labels, data_type)
    
    def _show_cluster_statistics(self, labels: np.ndarray, data_type: str):
        """Display cluster statistics"""
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        cluster_stats = pd.DataFrame({
            'Cluster ID': unique_labels,
            'Size': counts,
            'Percentage': (counts / len(labels) * 100).round(2)
        })
        
        cluster_stats = cluster_stats.sort_values('Size', ascending=False)
        
        st.subheader("Cluster Statistics")
        st.dataframe(cluster_stats.astype(str), use_container_width=True)
        
        # Visualize cluster sizes
        fig = px.bar(
            cluster_stats,
            x='Cluster ID',
            y='Size',
            title="Cluster Size Distribution",
            text='Size',
            color='Cluster ID',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        
        st.plotly_chart(fig, use_container_width=True)

class DataComparison:
    """
    Compare basic statistical features across datasets
    """

    @staticmethod
    def compare_datasets(processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        metric_columns = {
            'heart_rate': 'heart_rate',
            'sleep': 'duration_minutes',
            'steps': 'step_count'
        }

        comparison = {}

        for data_type, df in processed_data.items():
            if data_type not in metric_columns:
                continue

            col = metric_columns[data_type]
            if col not in df.columns:
                continue

            series = df[col].dropna()

            comparison[data_type.replace('_', ' ').title()] = {
                'Mean': series.mean(),
                'Median': series.median(),
                'Std Deviation': series.std(),
                'Minimum': series.min(),
                'Maximum': series.max(),
                'Variance': series.var()
            }

        return pd.DataFrame(comparison).round(2)



class FeatureModelingPipeline:
    """Complete pipeline for Milestone 2: Feature Extraction and Modeling"""
    def __init__(self):
        self.feature_extractor = TSFreshFeatureExtractor(feature_complexity='efficient')
        self.trend_modeler = ProphetTrendModeler()
        self.clusterer = BehaviorClusterer()
        self.feature_matrices = {}
        self.all_reports = {}
    def run_complete_milestone2(self, processed_data: Dict[str, pd.DataFrame],
                               window_size: int = 60,
                               forecast_periods: int = 100,
                               clustering_method: str = 'kmeans',
                               n_clusters: int = 3) -> Dict:
        st.write("")
        st.divider()
        st.header("🔬Extraction and Modeling")
        results = {
            'feature_matrices': {},
            'forecasts': {},
            'cluster_labels': {},
            'reports': {}
        }
        for data_type, df in processed_data.items():
            st.subheader(f"📊 Processing {data_type.replace('_', ' ').title()}")
            # Step 1: Feature Extraction with TSFresh
            with st.expander(f"🔵 Step 1: Extract Features - {data_type}", expanded=True):
                feature_matrix, extraction_report = self.feature_extractor.extract_features(
                    df, data_type, window_size
                )
                if not feature_matrix.empty:
                    results['feature_matrices'][data_type] = feature_matrix
                    results['reports'][f'{data_type}_extraction'] = extraction_report
                    st.write("**Top Features by Variance:**")
                    top_features = self.feature_extractor.get_top_features(10)
                    if not top_features.empty:
                        top_features['Feature'] = top_features['Feature'].astype(str)
                    st.dataframe(top_features, use_container_width=True)
                    self.feature_extractor.visualize_feature_distributions(6)
            # Step 2: Trend Modeling with Prophet
            with st.expander(f"🟡 Step 2: Model Trends with Prophet - {data_type}", expanded=True):
                forecast, modeling_report = self.trend_modeler.fit_and_predict(
                    df, data_type, forecast_periods
                )
                if not forecast.empty:
                    results['forecasts'][data_type] = forecast
                    results['reports'][f'{data_type}_modeling'] = modeling_report
                    self.trend_modeler.visualize_forecast(data_type, df)
                    anomalies = self.trend_modeler.get_anomalies_from_residuals(
                        data_type, threshold_std=3.0
                    )
                    if not anomalies.empty:
                        st.write(f"**detected {len(anomalies)} potential anomalies**")
                        st.dataframe(anomalies.head(10), use_container_width=True)
            # Step 3: Clustering
            with st.expander(f"🟢 Step 3: Cluster Behavioral Patterns - {data_type}", expanded=True):
                if data_type in results['feature_matrices']:
                    feature_matrix = results['feature_matrices'][data_type]
                    labels, clustering_report = self.clusterer.cluster_features(
                        feature_matrix, data_type, clustering_method, n_clusters
                    )
                    if len(labels) > 0:
                        results['cluster_labels'][data_type] = labels
                        results['reports'][f'{data_type}_clustering'] = clustering_report
                        self.clusterer.reduce_and_visualize(
                            feature_matrix, data_type, method='pca'
                        )
            st.markdown("---")
        self._generate_milestone2_summary(results)
        return results
    def _generate_milestone2_summary(self, results: Dict):
        pass

def add_milestone2_ui(processed_data):
    """Add Milestone 2 UI section to the main app"""
    
    st.divider()
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1.5rem 0;">
            <h2 style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🚀 Milestone 2: Advanced ML Features
            </h2>
            <p style="color: #6c757d; font-size: 1.1rem; margin-top: 0.5rem;">
                Complete Pipeline: TSFresh Feature Extraction • Prophet Forecasting • Clustering Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration for Milestone 2
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Milestone 2 Configuration")
    
    window_size = st.sidebar.slider(
        "Feature Window Size (minutes)",
        min_value=10, max_value=120, value=60, step=10,
        help="Window size for TSFresh feature extraction"
    )
    
    forecast_periods = st.sidebar.slider(
        "Prophet Forecast Periods",
        min_value=50, max_value=500, value=100, step=50,
        help="Number of future periods to forecast"
    )
    
    clustering_method = st.sidebar.selectbox(
        "Clustering Method",
        options=['kmeans', 'dbscan'],
        help="Choose clustering algorithm"
    )
    
    if clustering_method == 'kmeans':
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2, max_value=10, value=3
        )
    else:
        n_clusters = 3  # Default for DBSCAN
    
    # Initialize pipeline
    if 'milestone2_pipeline' not in st.session_state:
        st.session_state.milestone2_pipeline = FeatureModelingPipeline()
    
    # Run pipeline button with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center; margin: 1rem 0;">
                <p style="font-size: 0.95rem; color: #212529; margin-bottom: 1rem;">
                    Ready to process your data? Click below to start the pipeline
                </p>
            </div>
        """, unsafe_allow_html=True)
        process_button = st.button(
            "🚀 Run Milestone 2 Pipeline",
            type="primary",
            use_container_width=True
        )
    st.markdown("<br>", unsafe_allow_html=True)
    
    if process_button:
            # Run complete pipeline
            results = st.session_state.milestone2_pipeline.run_complete_milestone2(
                processed_data=processed_data,
                window_size=window_size,
                forecast_periods=forecast_periods,
                clustering_method=clustering_method,
                n_clusters=n_clusters
            )
            
            st.session_state.milestone2_results = results
           
    
    # Get results from session state
    results = st.session_state.get('milestone2_results', {})
    
    # Milestone 2 results section removed
    # The detailed results display has been removed to clean up the UI


    
    # Display content based on button clicks - COMMENTED OUT
    # The detailed results display has been removed
    pass


# ============================================================================
# MILESTONE 3: ANOMALY DETECTION PIPELINE
# ============================================================================

class AnomalyDetectionPipeline:
    """
    Complete Milestone 3 pipeline integrating all anomaly detection methods.
    """
    
    def __init__(self):
        self.threshold_detector = ThresholdAnomalyDetector()
        self.residual_detector = ResidualAnomalyDetector(threshold_std=3.0)
        self.cluster_detector = ClusterAnomalyDetector()
        self.visualizer = AnomalyVisualizer()
        
        self.processed_data = {}
        self.anomaly_reports = {}
    
    def run_complete_milestone3(self,
                                preprocessed_data: Dict[str, pd.DataFrame],
                                prophet_forecasts: Optional[Dict] = None,
                                cluster_labels: Optional[Dict] = None,
                                feature_matrices: Optional[Dict] = None) -> Dict:
        """
        Run complete Milestone 3 anomaly detection pipeline.

        Args:
            preprocessed_data: Output from Milestone 1
            prophet_forecasts: Output from Milestone 2 Prophet modeling
            cluster_labels: Output from Milestone 2 clustering
            feature_matrices: Output from Milestone 2 feature extraction

        Returns:
            Dictionary containing all anomaly detection results
        """

           


        try:
            # =========================================
            # RUN ALL ANOMALY DETECTION METHODS (ONCE)
            # =========================================
            st.write("")
            st.write("")
            st.header("🚨Anomaly Detection")
            st.write("")
            st.markdown("---")
           

            results = {
                "reports": {},
                "data_with_anomalies": {}
            }


            for data_type, df in preprocessed_data.items():
                results["reports"][data_type] = {}

                # ---------- Method 1: Threshold ----------
                df_with_threshold, threshold_report = self.threshold_detector.detect_anomalies(
                    df, data_type
                )
                results["reports"][data_type]["threshold"] = threshold_report

                # ---------- Method 2: Prophet Residual ----------
                if prophet_forecasts and data_type in prophet_forecasts:
                    forecast = prophet_forecasts[data_type]
                    df_with_residual, residual_report = self.residual_detector.detect_anomalies_from_prophet(
                        df_with_threshold, forecast, data_type
                    )
                    results["reports"][data_type]["residual"] = residual_report
                    df_final = df_with_residual
                else:
                    df_final = df_with_threshold

                # ---------- Method 3: Cluster ----------
                if (
                    cluster_labels
                    and data_type in cluster_labels
                    and feature_matrices
                    and data_type in feature_matrices
                ):
                    labels = cluster_labels[data_type]
                    features = feature_matrices[data_type]

                    _, cluster_report = self.cluster_detector.detect_cluster_outliers(
                        features, labels, data_type
                    )
                    results["reports"][data_type]["cluster"] = cluster_report

                results["data_with_anomalies"][data_type] = df_final
                self.processed_data[data_type] = df_final




            # =========================================
            # VISUALIZATIONS (PER DATA TYPE ONLY)
            # =========================================


        except Exception as e:
            st.error(f"❌ Error in Milestone 3 pipeline: {str(e)}")
            st.exception(e)
            return {}

        return results


    def display_milestone3_results(self, results: Dict):
        """
        Display the results of Milestone 3 anomaly detection.
        Since results are shown immediately during processing, this shows a summary.
        """
        try:
            if not results or 'data_with_anomalies' not in results:
                st.error("❌ No results to display - results dictionary is empty or missing data_with_anomalies key")
                return

            if not results['data_with_anomalies']:
                st.error("❌ No anomaly data found in results")
                return

           
            st.markdown("**Complete analysis results overview**")

            # Clean summary metrics
            total_data_types = len(results['data_with_anomalies'])
            total_anomalies = 0

            # Calculate total anomalies across all data types (only boolean anomaly flags)
            for data_type, df in results['data_with_anomalies'].items():
                if df is not None and not df.empty:
                    anomaly_flag_cols = [col for col in df.columns if col.endswith('_anomaly') and df[col].dtype == bool]
                    for col in anomaly_flag_cols:
                        total_anomalies += df[col].sum()

            # Summary cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Data Types Analyzed", total_data_types)
            with col2:
                st.metric("🚨 Total Anomalies Detected", total_anomalies)
            with col3:
                avg_anomalies = total_anomalies / total_data_types if total_data_types > 0 else 0
                st.metric("📈 Avg Anomalies per Type", f"{avg_anomalies:.1f}")

            st.markdown("---")

            # Show detailed results for each data type
            for data_type, df in results['data_with_anomalies'].items():
                if df is None or df.empty:
                    st.warning(f"⚠️ No data available for {data_type}")
                    continue

                # Clean data type header with anomaly count
                anomaly_flag_cols = [col for col in df.columns if col.endswith('_anomaly') and df[col].dtype == bool]
                data_type_anomalies = sum(df[col].sum() for col in anomaly_flag_cols)

                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                                padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #667eea;">
                        <h4 style="color: #667eea; margin: 0; font-size: 1.2rem; font-weight: 600;">
                            📊 {data_type.replace('_', ' ').title()}
                        </h4>
                        <p style="color: #495057; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                            {len(df)} records • {data_type_anomalies} anomalies detected
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Quick summary of detection methods
                method_cols = st.columns(3)
                methods = ['threshold', 'residual', 'cluster']

                for i, method in enumerate(methods):
                    with method_cols[i]:
                        if 'reports' in results and data_type in results['reports'] and method in results['reports'][data_type]:
                            report = results['reports'][data_type][method]
                            count = report.get('anomalies_detected', 0)
                            st.metric(f"{method.title()} Anomalies", count)
                        else:
                            st.metric(f"{method.title()} Anomalies", "N/A")

                # Show visualization directly after metrics
                try:
                    if data_type == 'heart_rate':
                        self.visualizer.plot_heart_rate_anomalies(df, title=f"Heart Rate Anomalies")
                    elif data_type == 'steps':
                        self.visualizer.plot_steps_anomalies(df, title=f"Step Count Anomalies")
                    elif data_type == 'sleep':
                        self.visualizer.plot_sleep_anomalies(df, title=f"Sleep Pattern Anomalies")
                    else:
                        st.warning(f"No visualization available for {data_type}")
                except Exception as viz_error:
                    st.error(f"❌ Visualization error: {str(viz_error)}")
                    # Fallback: show basic dataframe
                    st.write("**Raw Data with Anomalies:**")
                    st.dataframe(df.head(20), use_container_width=True)

            # Generate comprehensive summary
            st.markdown("---")
            try:
                self.visualizer.create_anomaly_summary_dashboard(results['reports'])
            except Exception as summary_error:
                st.error(f"❌ Error creating summary dashboard: {str(summary_error)}")

            # Export functionality
            try:
                self.visualizer.export_anomaly_report(results)
            except Exception as export_error:
                st.error(f"❌ Error creating export functionality: {str(export_error)}")

        except Exception as e:
            st.error(f"❌ Error displaying Milestone 3 results: {str(e)}")
            st.exception(e)
def add_milestone3_ui(processed_data):
    """Add Milestone 3 UI section to the main app"""
    
    st.divider()
    st.markdown("""
        <div style="text-align: center; margin: 3rem 0 2rem 0;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #1e3a8a;">
                🚨 Milestone 3: Anomaly Detection
            </h2>
            <p style="color: #6c757d; font-size: 1.1rem; margin-top: 0.5rem;">
                Detect unusual health patterns using advanced AI algorithms
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration for Milestone 3
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 Milestone 3 Configuration")
    
    threshold_std = st.sidebar.slider(
        "Anomaly Sensitivity (σ)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Lower values = more sensitive detection"
    )
    
    # Initialize pipeline
    if 'milestone3_pipeline' not in st.session_state:
        st.session_state.milestone3_pipeline = AnomalyDetectionPipeline()
    
    # Run pipeline button with enhanced styling
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center; margin: 1rem 0;">
                <p style="font-size: 0.95rem; color: #212529; margin-bottom: 1rem;">
                    Ready to process your data? Click below to start the pipeline
                </p>
            </div>
        """, unsafe_allow_html=True)
        process_button = st.button(
            "🚨 Run Milestone 3 Pipeline",
            type="primary",
            use_container_width=True,
            key="milestone3_run_button"
        )
    st.markdown("<br>", unsafe_allow_html=True)
    
    if process_button:
        # Clear any previous results to avoid confusion
        if 'milestone3_results' in st.session_state:
            del st.session_state.milestone3_results

        # Show processing message immediately
        with st.spinner("🔄 Running Milestone 3 anomaly detection pipeline..."):
            try:
                st.info("🔍 Starting anomaly detection analysis...")

                # Get Milestone 2 results if available
                prophet_forecasts = None
                cluster_labels = None
                feature_matrices = None

                if 'milestone2_results' in st.session_state:
                    m2_results = st.session_state.milestone2_results
                    prophet_forecasts = m2_results.get('forecasts', {})
                    cluster_labels = m2_results.get('cluster_labels', {})
                    feature_matrices = m2_results.get('feature_matrices', {})
                    st.info("✅ Using Milestone 2 results for enhanced anomaly detection")
                else:
                    st.info("ℹ️ Milestone 2 results not available. Running basic anomaly detection.")

                # Update threshold
                st.session_state.milestone3_pipeline.residual_detector.threshold_std = threshold_std

                # Validate input data
                if not processed_data:
                    st.error("❌ No processed data available. Please complete Milestone 1 first.")
                    return

                st.info(f"📊 Processing {len(processed_data)} data types: {list(processed_data.keys())}")

                # Run pipeline (processing only)
                results = st.session_state.milestone3_pipeline.run_complete_milestone3(
                    preprocessed_data=processed_data,
                    prophet_forecasts=prophet_forecasts,
                    cluster_labels=cluster_labels,
                    feature_matrices=feature_matrices
                )

                # Validate results
                if results and 'data_with_anomalies' in results and results['data_with_anomalies']:
                    st.session_state.milestone3_results = results
                    st.success("✅ Milestone 3 Pipeline Completed Successfully!")

                    # Count anomalies detected
                    total_anomalies = 0
                    for data_type, df in results['data_with_anomalies'].items():
                        if 'anomaly_score' in df.columns:
                            try:
                                if df['anomaly_score'].dtype in ['int64', 'float64']:
                                    anomaly_count = (df['anomaly_score'] > 0).sum()
                                elif df['anomaly_score'].dtype == 'object':
                                    # For string columns, count non-null values
                                    anomaly_count = df['anomaly_score'].notna().sum()
                                else:
                                    # Try to convert to numeric
                                    anomaly_count = (pd.to_numeric(df['anomaly_score'], errors='coerce') > 0).sum()
                                total_anomalies += anomaly_count
                                st.info(f"🔍 {data_type}: {anomaly_count} anomalies detected")
                            except Exception as e:
                                st.warning(f"⚠️ Could not count anomalies for {data_type}: {e}")
                                anomaly_count = 0

                else:
                    st.error("❌ Failed to run Milestone 3 pipeline. No valid results generated.")
                    st.error("Check that Milestone 1 and Milestone 2 have been completed successfully.")

            except Exception as e:
                st.error(f"❌ Error running Milestone 3 pipeline: {str(e)}")
                st.exception(e)
                st.error("Please check the error details above and try again.")


def show_milestone2_results(milestone2_results):
    """Display Milestone 2 results persistently"""
    if not milestone2_results:
        return
        
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                    padding: 2rem; border-radius: 20px; border-left: 6px solid #667eea; 
                    text-align: center; margin: 2rem 0; box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);">
            <h3 style="color: #667eea; font-weight: 800; font-size: 1.8rem; margin-bottom: 0.5rem;">
                            ✅ Extraction Completed!
            </h3>
            <p style="color: #495057; font-size: 1.1rem; margin: 0;">
                Advanced ML features extracted and models trained successfully
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display results summary
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1.5rem 0;">
            <h2 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Summary
            </h2>
            
        </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2d3748, #1a202c); padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">Data Types Processed</div>
                <div style="color: white; font-size: 2.5rem; font-weight: 800;">{len(milestone2_results['feature_matrices'])}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_features = sum(len(df.columns) for df in milestone2_results['feature_matrices'].values())
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4a5568, #2d3748); padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">Total Features Extracted</div>
                <div style="color: white; font-size: 2.5rem; font-weight: 800;">{total_features}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #718096, #4a5568); padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">Prophet Models Trained</div>
                <div style="color: white; font-size: 2.5rem; font-weight: 800;">{len(milestone2_results['forecasts'])}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a0aec0, #718096); padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">Clustering Complete</div>
                <div style="color: white; font-size: 2.5rem; font-weight: 800;">{len(milestone2_results['cluster_labels'])}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed results in tabs
    st.markdown("### 📊 Detailed Results")
    
    tabs = st.tabs(["🔵 Feature Extraction", "🟡 Prophet Forecasting", "🟢 Clustering Analysis"])
    
    with tabs[0]:
        st.subheader("TSFresh Feature Extraction Results")
        for data_type, features_df in milestone2_results['feature_matrices'].items():
            with st.expander(f"📈 {data_type.replace('_', ' ').title()} Features ({len(features_df.columns)} features)", expanded=False):
                st.dataframe(features_df.head().astype(str), use_container_width=True)
                st.info(f"Extracted {len(features_df.columns)} statistical features from {len(features_df)} time windows")
    
    with tabs[1]:
        st.subheader("Prophet Forecasting Results")
        for data_type, forecast_df in milestone2_results['forecasts'].items():
            with st.expander(f"📈 {data_type.replace('_', ' ').title()} Forecast", expanded=False):
                if not forecast_df.empty:
                    st.dataframe(forecast_df.head().astype(str), use_container_width=True)
                    st.success(f"Generated {len(forecast_df)} forecast points")
                    
                    # Get performance metrics from reports
                    modeling_report_key = f'{data_type}_modeling'
                    if modeling_report_key in milestone2_results['reports']:
                        perf = milestone2_results['reports'][modeling_report_key]
                        if 'mae' in perf and 'rmse' in perf:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("MAE", f"{perf.get('mae', 0):.3f}")
                            with col_b:
                                st.metric("RMSE", f"{perf.get('rmse', 0):.3f}")
                        if 'mape' in perf:
                            st.metric("MAPE", f"{perf.get('mape', 0):.2f}%")
    
    with tabs[2]:
        st.subheader("Clustering Analysis Results")
        for data_type, labels in milestone2_results['cluster_labels'].items():
            with st.expander(f"📈 {data_type.replace('_', ' ').title()} Clusters", expanded=False):
                unique_labels = len(set(labels))
                st.metric("Number of Clusters", unique_labels)
                
                # Show cluster distribution
                import pandas as pd
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                st.bar_chart(cluster_counts)


def show_milestone1_results(processed_data):
    """Display Milestone 1 results persistently"""
    if not processed_data:
        return
        
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(32, 201, 151, 0.1)); 
                    padding: 1.5rem; border-radius: 20px; border-left: 6px solid #28a745; 
                    text-align: center; margin: 1rem 0; box-shadow: 0 8px 24px rgba(40, 167, 69, 0.15);">
            <h3 style="color: #28a745; font-weight: 800; font-size: 1.8rem; margin-bottom: 0.5rem;">
                ✅ Pipeline Completed Successfully!
            </h3>
            <p style="color: #495057; font-size: 1.1rem; margin: 0;">
                Your data has been processed and is ready for analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Visualizations
    # st.markdown("""
    #     <div style="text-align: center; margin: 1rem 0 1rem 0;">
    #         <h2 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); 
    #                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    #             📊 Data Visualization & Insights
    #         </h2>
    #         <p style="color: #6c757d; font-size: 1rem; margin-top: 0.5rem;">
    #             Explore your processed data with interactive charts and statistics
    #         </p>
    #     </div>
    # """, unsafe_allow_html=True)

    #TAB SWITCH
    #tabs = st.tabs([f"📈 {dt.replace('_', ' ').title()}" 
       #            for dt in processed_data.keys()])
    
    #  for idx, (data_type, df) in enumerate(processed_data.items()):
    #     with tabs[idx]:
    #         create_visualization(df, data_type)
            
    #         with st.expander("🔍 View Processed Data"):
    #            st.dataframe(df.head(100), use_container_width=True)
    # ========== TASK 2: FORMAT COMPARISON ==========
    # st.divider()
    # st.markdown("""
    #     <div style="text-align: center; margin: 1rem 0 1rem 0;">
    #         <h2 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #f093fb, #f5576c); 
    #                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    #             📊 Task 2: Format Comparison Analysis
    #         </h2>
    #         <p style="color: #6c757d; font-size: 1rem; margin-top: 0.5rem;">
    #             Performance metrics and comparison across different data formats
    #         </p>
    #     </div>
    # """, unsafe_allow_html=True)
    
    # with st.expander("📈 File Format Performance Comparison", expanded=False):
    #     # Create comparison
    #     formats_to_compare = {}
    #     for data_type, df in processed_data.items():
    #         formats_to_compare[f"{data_type} (processed)"] = df
        
    #     comparison_df = compare_file_formats(formats_to_compare)
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         st.subheader("📋 Format Metrics")
    #         st.dataframe(comparison_df, use_container_width=True)
            
    #         st.info("""
    #         **Metrics Explained:**
    #         - **Memory (MB)**: RAM usage for storing data
    #         - **Load Time (s)**: Time to copy/process data
    #         - **Avg Row Size**: Memory per record
    #         """)
        
    #     with col2:
    #         st.subheader("📊 Performance Visualization")
    #         fig = create_format_comparison_chart(comparison_df)
    #         st.plotly_chart(fig, use_container_width=True)

    # Export section
    st.divider()
    export_processed_data(processed_data)
    
    # ========== TIMESTAMP VALIDATION DASHBOARD ==========
 
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0 1rem 0;">
           <h2 style="font-size: 2.2rem; font-weight: 800; color: black;">
            🕒 Timestamp Validation 
           </h2>

        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    for data_type, df in processed_data.items():
        with st.expander(f"📅 {data_type.replace('_', ' ').title()} - Timestamp Analysis", expanded=False):
            
            st.subheader(f"24-Hour Data Coverage Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # 24-hour heatmap
                    fig_heatmap = create_timestamp_heatmap(df, 'timestamp')
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    st.caption("📊 Shows data distribution across days and hours")
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")
            
            with col2:
                try:
                    # Timezone distribution
                    fig_tz = create_timezone_distribution(df, 'timestamp')
                    st.plotly_chart(fig_tz, use_container_width=True)
                    
                    st.caption("📊 Shows hourly record distribution")
                except Exception as e:
                    st.error(f"Error creating distribution: {str(e)}")
            
            # Timestamp statistics
            st.subheader("📈 Timestamp Statistics")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Start Time", df['timestamp'].min().strftime('%Y-%m-%d %H:%M'))
            with col_b:
                st.metric("End Time", df['timestamp'].max().strftime('%Y-%m-%d %H:%M'))
            with col_c:
                time_span = df['timestamp'].max() - df['timestamp'].min()
                st.metric("Time Span", f"{time_span.days}d {time_span.seconds//3600}h")
            with col_d:
                if df['timestamp'].dt.tz is not None:
                    tz_info = str(df['timestamp'].dt.tz)
                else:
                    tz_info = "Naive (no TZ)"
                st.metric("Timezone", tz_info)

   
    
    # Download session report
    # st.markdown("### 💾 Session Report")
    # session_report = {
    #     'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
    #     'timestamp': datetime.now().isoformat(),
    #     'datasets_processed': len(processed_data),
    #     'total_records': sum(len(df) for df in processed_data.values()),
    #     'anomalies_detected': sum(df.get('anomaly', pd.Series(dtype=bool)).sum() if 'anomaly' in df.columns else 0 for df in processed_data.values()),
    #     'average_quality_score': sum(st.session_state.pipeline.reports.get(f'{dt}_validation', {}).get('metrics', {}).get('quality_score', 0) for dt in processed_data.keys()) / len(processed_data) if processed_data else 0,
    #     'processing_settings': {
    #         'target_frequency': getattr(st.session_state, 'target_freq', 'N/A'),
    #         'fill_method': getattr(st.session_state, 'fill_method', 'N/A')
    #     },
    #     'datasets': summary_items
    # }
    
    # report_json = json.dumps(session_report, indent=2)
    # st.download_button(
    #     label="📥 Download Session Report (JSON)",
    #     data=report_json,
    #     file_name=f"fitpulse_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    #     mime="application/json",
    #     use_container_width=True
    # )


def main():
    """Main Streamlit application"""
    
    # Initialize ALL session state variables at the very beginning
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = PreprocessingPipeline()
    
    # Apply theme immediately after initialization
    if st.session_state.theme == 'dark':
        st.markdown("""
            <style>
            /* Dark theme - Complete override */
            .stApp {
                background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%) !important;
            }
            .stApp > header {
                background-color: transparent !important;
            }
            .main .block-container {
                background-color: transparent !important;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a1d29 0%, #0e1117 100%) !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                background-color: transparent !important;
            }
            
            /* Text colors */
            .stApp, .stApp * {
                color: #fafafa !important;
            }
            .stMarkdown, .stText, p, span, div, label {
                color: #fafafa !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #fafafa !important;
            }
            
            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
            }
            
            /* Metrics */
            .stMetric {
                background: rgba(38, 39, 48, 0.9) !important;
                border: 1px solid #3d3d4d !important;
            }
            .stMetric label {
                color: #b8b9c1 !important;
            }
            .stMetric [data-testid="stMetricValue"] {
                color: #fafafa !important;
            }
            
            /* Input fields */
            .stSelectbox > div > div, .stTextInput > div > div {
                background-color: #262730 !important;
                color: #fafafa !important;
                border-color: #3d3d4d !important;
            }
            
            /* Dataframes */
            .dataframe {
                background-color: #262730 !important;
                color: #fafafa !important;
            }
            .dataframe thead tr th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
            }
            .dataframe tbody tr {
                background-color: #1a1d29 !important;
            }
            .dataframe tbody tr:hover {
                background-color: #262730 !important;
            }
            
            /* Expanders */
            .streamlit-expanderHeader {
                background-color: #262730 !important;
                border-color: #3d3d4d !important;
                color: #fafafa !important;
            }
            
            /* Info/Warning/Success boxes */
            .stAlert {
                background-color: rgba(38, 39, 48, 0.8) !important;
                color: #fafafa !important;
            }
            
            /* Dividers */
            hr {
                border-color: #3d3d4d !important;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: transparent !important;
                border-bottom-color: #3d3d4d !important;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #262730 !important;
                color: #fafafa !important;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background-color: #262730 !important;
                border-color: #667eea !important;
            }
            [data-testid="stFileUploader"] section {
                background-color: #1a1d29 !important;
                border-color: #667eea !important;
            }
            [data-testid="stFileUploader"] section > div {
                background-color: #262730 !important;
                color: #fafafa !important;
            }
            [data-testid="stFileUploader"] label {
                color: #fafafa !important;
            }
            [data-testid="stFileUploader"] small {
                color: #b8b9c1 !important;
            }
            [data-testid="stFileUploader"] button {
                background-color: #667eea !important;
                color: white !important;
            }
            
            /* Spinner */
            .stSpinner > div {
                border-top-color: #667eea !important;
            }
            
            /* Checkbox */
            .stCheckbox {
                color: #fafafa !important;
            }
            
            /* Radio */
            .stRadio label {
                color: #fafafa !important;
            }
            
            /* Download button */
            .stDownloadButton > button {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    
    st.markdown("""
        <style>
        /* Import modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
     .stApp {
    background: #e8f0f7;
}
        /* Main container with glassmorphism */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* Animated header with premium gradient */
        .main-header {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1.5rem 0;
            animation: gradientShift 8s ease infinite, fadeIn 0.8s ease-out;
            letter-spacing: -0.02em;
            text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        /* Premium metric cards with glassmorphism */
        .stMetric {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.8rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            animation: slideIn 0.6s ease-out;
        }
        
        .stMetric::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.5);
        }
        
        .stMetric::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .stMetric:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
            border-color: rgba(102, 126, 234, 0.5);
        }
        
        .stMetric:hover::after {
            opacity: 1;
        }
        
        /* Premium buttons with advanced effects */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 2.5rem;
            font-weight: 700;
            font-size: 1.1rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
            letter-spacing: 0.02em;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(-1px) scale(1.02);
        }
        
        /* Styled expanders with premium look */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 14px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            font-weight: 600;
            padding: 1.2rem 1.8rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            transform: translateX(5px);
        }
        
        /* Beautiful dataframes */
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        .dataframe thead tr th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 1.2rem !important;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        
        .dataframe tbody tr {
            transition: all 0.2s ease;
        }
        
        .dataframe tbody tr:hover {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%) !important;
            transform: scale(1.01);
        }
        
        /* Modern tabs with smooth transitions */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 3px solid rgba(102, 126, 234, 0.2);
            padding-bottom: 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 14px 14px 0 0;
            padding: 1rem 2rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid transparent;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.1);
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            transform: translateY(-4px);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        /* Sidebar with premium styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            animation: slideIn 0.6s ease-out;
        }
        
        /* File uploader with drag-and-drop emphasis */
        [data-testid="stFileUploader"] {
            border: 3px dashed rgba(102, 126, 234, 0.4);
            border-radius: 20px;
            padding: 2.5rem;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(5px);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.05);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
            transform: scale(1.01);
        }
        
        /* Enhanced alert messages */
        .stSuccess {
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(32, 201, 151, 0.1) 100%);
            border-left: 5px solid #28a745;
            border-radius: 14px;
            padding: 1.2rem 1.8rem;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
        }
        
        .stInfo {
            background: linear-gradient(135deg, rgba(23, 162, 184, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
            border-left: 5px solid #17a2b8;
            border-radius: 14px;
            padding: 1.2rem 1.8rem;
            box-shadow: 0 4px 12px rgba(23, 162, 184, 0.15);
        }
        
        .stWarning {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
            border-left: 5px solid #ffc107;
            border-radius: 14px;
            padding: 1.2rem 1.8rem;
            box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        }
        
        .stError {
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
            border-left: 5px solid #dc3545;
            border-radius: 14px;
            padding: 1.2rem 1.8rem;
            box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        }
        
        /* Premium download buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white !important;
            border-radius: 12px;
            padding: 0.75rem 1.8rem;
            font-weight: 700;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
            border: none;
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        }
        
        /* Animated dividers */
        hr {
            margin: 2.5rem 0;
            border: none;
            height: 3px;
            background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 50%, #f093fb 80%, transparent 100%);
            border-radius: 2px;
            animation: shimmer 3s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        /* Charts with premium styling */
        .js-plotly-plot {
            border-radius: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            background: white;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .js-plotly-plot:hover {
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
            transform: translateY(-2px);
        }
        
        /* Spinner customization */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            border-radius: 12px;
            border-color: rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }
        
        /* Checkbox styling */
        .stCheckbox {
            padding: 0.5rem;
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        
        .stCheckbox:hover {
            background: rgba(102, 126, 234, 0.05);
        }
        
        /* Code blocks */
        code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(102, 126, 234, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            font-size: 0.9em;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.05);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # HEADER SECTION with Logo
    col_title, col_logo = st.columns([7, 1])
    
    with col_title:
        st.markdown("<h1 style='text-align: center; font-weight: 800; font-size: 3rem; color: #1e3a8a; margin: 1rem 0;'>FitPulse: AI-Based Wellness Anomaly Detection</h1>", unsafe_allow_html=True)
    
    with col_logo:
        try:
            from PIL import Image
            logo_img = Image.open("logo.png")
            # Add CSS to make it circular
            st.markdown("""
                <style>
                [data-testid="column"]:last-child img {
                    border-radius: 50%;
                    border: 3px solid #1e3a8a;
                    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
                }
                </style>
            """, unsafe_allow_html=True)
            st.image(logo_img, width=90)
        except Exception as e:
            st.markdown("""
                <div style="width: 90px; height: 90px; border-radius: 50%; 
                     background: linear-gradient(135deg, #1e3a8a, #3b82f6);
                     display: flex; align-items: center; justify-content: center;
                     box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3); border: 3px solid #1e3a8a;">
                    <span style="font-size: 2.5rem;">🏥</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Subtitle and feature badges
    st.markdown('''
        <div style="text-align: center; margin: 2rem 0 3rem 0;">
            <p style="font-size: 1.1rem; color: #212529; font-weight: 500; margin-bottom: 1.5rem;">
               Secure, enterprise-grade health data analytics enhanced by AI-powered insights.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.5rem 1.2rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">
                    ✓ Multi-Format Support
                </span>
                <span style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; padding: 0.5rem 1.2rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);">
                    ✓ Smart Validation
                </span>
                <span style="background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; padding: 0.5rem 1.2rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);">
                    ✓ AI-Powered Analytics
                </span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    st.divider()
    
    
    with st.sidebar:
        # Theme Toggle
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("☀️ Light", use_container_width=True, type="primary" if st.session_state.theme == 'light' else "secondary"):
                st.session_state.theme = 'light'
        with theme_col2:
            if st.button("🌙 Dark", use_container_width=True, type="primary" if st.session_state.theme == 'dark' else "secondary"):
                st.session_state.theme = 'dark'
        
        st.divider()
        
        st.markdown("### 📊 Data Source")
        use_sample = st.checkbox("🧪 Use Sample Data", value=True, 
                                 help="Toggle to test with synthetic data")
        
        st.divider()
        
        st.markdown("### 🔧 Processing Settings")
        
        st.markdown("**⏱️ Target Frequency**")
        target_freq = st.selectbox(
            "Resample interval:",
            options=['30 sec', '1 min', '5 min', '15 min', '30 min', '1 hour'],
            index=1,  # Back to 1 min default, but smart logic prevents excessive expansion
            help="Resample data to this uniform time interval",
            label_visibility="collapsed"
        )
        
        st.markdown("**🔄 Gap Filling Method**")
        fill_method = st.selectbox(
            "Fill strategy:",
            options=['interpolate', 'forward_fill', 'backward_fill', 'zero'],
            index=0,
            help="Method to fill missing values after resampling",
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### ℹ️ About")
        st.info("""
        **Key Features:**
        
        🔹 Multi-format support  
        🔹 Data validation  
        🔹 Anomaly detection  
        🔹 AI insights  
        🔹 Interactive charts
        """)
        
        st.divider()
       
    
    # Main content area
    # Pipeline already initialized at the beginning of main()
    
    # File upload or sample data with premium styling
    uploaded_files = None
    
    if not use_sample:
        st.markdown("""
            <div style="text-align: center; margin: 2rem 0 1rem 0;">
                <h2 style="font-size: 2rem; font-weight: 700; color: #495057; margin-bottom: 0.5rem;">
                    📁 Upload Your Fitness Data
                </h2>
                <p style="color: #6c757d; font-size: 1rem;">
                    Drag and drop your files or click to browse
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose fitness tracker files (CSV, JSON, or Excel)",
            type=['csv', 'json', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload heart rate, sleep, or step data files",
            label_visibility="collapsed"
        )

        # Check if uploaded files have content
        valid_uploaded_files = []
        if uploaded_files:
            for file in uploaded_files:
                if hasattr(file, 'size') and file.size > 0:
                    valid_uploaded_files.append(file)
                elif hasattr(file, 'getvalue'):  # StringIO objects
                    content = file.getvalue()
                    if content and len(content.strip()) > 0:
                        valid_uploaded_files.append(file)

        # Use uploaded files if they have content, otherwise use sample data
        if valid_uploaded_files:
            st.success(f"✅ Loaded {len(valid_uploaded_files)} uploaded file(s)")
            uploaded_files = valid_uploaded_files
        else:
            st.info("📁 **No valid files uploaded**: Using CSV files from the sample_data folder.")
            uploaded_files = load_sample_csv_files()
    else:
        st.markdown(""" 
    <div style="background: linear-gradient(135deg, #003d6b 0%, #00509e 100%);  
                padding: 1.25rem; 
                border-radius: 16px; 
                box-shadow: 0 6px 24px rgba(0, 61, 107, 0.25);
                text-align: center; 
                margin: 1.5rem 0;
                position: relative;
                overflow: hidden;"> 
        <div style="position: absolute; top: -50%; right: -50%; width: 200%; height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                    animation: pulse 3s ease-in-out infinite;"></div>
        <div style="position: relative; z-index: 1;">
            <p style="margin: 0; 
                      font-size: 1.5rem; 
                      margin-bottom: 0.3rem;"> 
                🧪
            </p>
            <p style="margin: 0; 
                      font-size: 1.1rem; 
                      font-weight: 700; 
                      color: #ffffff;
                      text-shadow: 0 2px 4px rgba(0,0,0,0.1);
                      letter-spacing: 0.3px;"> 
                Sample Data Mode Activated
            </p> 
            <p style="margin: 0.5rem auto 0 auto; 
                      font-size: 0.9rem; 
                      color: rgba(255, 255, 255, 0.9);
                      max-width: 450px;
                      line-height: 1.5;"> 
                Realistic synthetic health data is loaded and ready for analysis
            </p>
            <div style="margin-top: 1rem; 
                        display: inline-block; 
                        padding: 0.4rem 1.2rem; 
                        background: rgba(255, 255, 255, 0.2); 
                        border-radius: 20px;
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(255, 255, 255, 0.3);">
                <span style="color: #fff; font-size: 0.85rem; font-weight: 600;">
                    ✓ Demo Environment
                </span>
            </div>
        </div>
    </div> 
    <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.5; }
            50% { transform: scale(1.1) rotate(180deg); opacity: 0.3; }
        }
    </style>
""", unsafe_allow_html=True)
        uploaded_files = load_sample_csv_files()
    
    st.markdown("<br>", unsafe_allow_html=True)
    process_button = st.button(
        "🚀 Run Complete Pipeline",
        type="primary",
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if process_button:
        if uploaded_files:
            with st.spinner("🔄 Processing data through all milestones..."):
                try:
                    # Milestone 1
                    processed_data = st.session_state.pipeline.run(
                        uploaded_files,
                        target_freq,
                        fill_method
                    )
                    if not processed_data:
                        st.error("❌ Pipeline failed to process data. Please check your input files.")
                        return
                    st.session_state.processed_data = processed_data
                
                    show_milestone1_results(processed_data)

                    # Milestone 2
                    if 'milestone2_pipeline' not in st.session_state:
                        st.session_state.milestone2_pipeline = FeatureModelingPipeline()
                    m2_results = st.session_state.milestone2_pipeline.run_complete_milestone2(
                        processed_data=processed_data,
                        window_size=st.session_state.get('window_size', 60),
                        forecast_periods=st.session_state.get('forecast_periods', 100),
                        clustering_method=st.session_state.get('clustering_method', 'kmeans'),
                        n_clusters=st.session_state.get('n_clusters', 3)
                    )
                    st.session_state.milestone2_results = m2_results
                    # st.success("✅ Milestone 2 Pipeline Complete!")
                    show_milestone2_results(m2_results)

                    # Milestone 3
                    if 'milestone3_pipeline' not in st.session_state:
                        st.session_state.milestone3_pipeline = AnomalyDetectionPipeline()
                    m3_results = st.session_state.milestone3_pipeline.run_complete_milestone3(
                        preprocessed_data=processed_data,
                        prophet_forecasts=m2_results.get('forecasts'),
                        cluster_labels=m2_results.get('cluster_labels'),
                        feature_matrices=m2_results.get('feature_matrices')
                    )
                    st.session_state.milestone3_results = m3_results
                    st.success("✅ Anomaly Detection Complete!")
                    st.session_state.milestone3_pipeline.display_milestone3_results(m3_results)
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please upload files or enable sample data.")


if __name__ == "__main__":
    main()
