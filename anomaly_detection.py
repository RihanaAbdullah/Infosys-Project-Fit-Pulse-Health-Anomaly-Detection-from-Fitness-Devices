"""
Anomaly Detection Module for FitPulse
Contains all anomaly detection algorithms and methods
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class ThresholdAnomalyDetector:
    """
    Rule-based anomaly detection using configurable thresholds.
    Detects when metrics exceed predefined limits for a sustained period.
    """
    
    def __init__(self):
        self.threshold_rules = {
            'heart_rate': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,  # Below this is too low (bradycardia)
                'max_threshold': 120,  # Above this is too high (tachycardia at rest)
                'sustained_minutes': 10,  # Must persist for this duration
                'description': 'Heart rate outside normal resting range'
            },
            'steps': {
                'metric_name': 'step_count',
                'min_threshold': 0,
                'max_threshold': 1000,  # More than 1000 steps/min is unrealistic
                'sustained_minutes': 5,
                'description': 'Unrealistic step count detected'
            },
            'sleep': {
                'metric_name': 'duration_minutes',
                'min_threshold': 180,  # Less than 3 hours of sleep
                'max_threshold': 720,  # More than 12 hours of sleep
                'sustained_minutes': 0,  # Applies to daily totals
                'description': 'Unusual sleep duration'
            }
        }
        
        self.detected_anomalies = []
    
    def detect_anomalies(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect threshold-based anomalies in the data.
        
        Args:
            df: DataFrame with timestamp and metric columns
            data_type: Type of data ('heart_rate', 'steps', 'sleep')
            
        Returns:
            DataFrame with anomaly flags and detection report
        """
        
        
        report = {
            'method': 'Threshold-Based',
            'data_type': data_type,
            'total_records': len(df),
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
            'threshold_info': {}
        }
        
        if data_type not in self.threshold_rules:
            st.warning(f"No threshold rules defined for {data_type}")
            return df, report
        
        rule = self.threshold_rules[data_type]
        metric_col = rule['metric_name']
        
        if metric_col not in df.columns:
            st.error(f"Metric column '{metric_col}' not found in data")
            return df, report
        
        df_result = df.copy()
        df_result['threshold_anomaly'] = False
        df_result['anomaly_reason'] = ''
        df_result['severity'] = 'Normal'
        
        # Detect threshold violations
        too_high = df_result[metric_col] > rule['max_threshold']
        too_low = df_result[metric_col] < rule['min_threshold']

        df_result['threshold_violation'] = too_high | too_low
        
        # Apply sustained duration filter (for heart rate and steps)
        if rule['sustained_minutes'] > 0:
            # Convert minutes to number of records (assuming 1-min frequency)
            window_size = rule['sustained_minutes']
            
            # Rolling window to check if anomaly persists
            too_high_sustained = too_high.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            too_low_sustained = too_low.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            
            df_result.loc[too_high_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_high_sustained, 'anomaly_reason'] = f'High {metric_col} (>{rule["max_threshold"]})'
            df_result.loc[too_high_sustained, 'severity'] = 'High'
            
            df_result.loc[too_low_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_low_sustained, 'anomaly_reason'] = f'Low {metric_col} (<{rule["min_threshold"]})'
            df_result.loc[too_low_sustained, 'severity'] = 'Medium'
        else:
            # Instant detection (for sleep duration)
            df_result.loc[too_high, 'threshold_anomaly'] = True
            df_result.loc[too_high, 'anomaly_reason'] = f'Excessive {metric_col}'
            df_result.loc[too_high, 'severity'] = 'Medium'
            
            df_result.loc[too_low, 'threshold_anomaly'] = True
            df_result.loc[too_low, 'anomaly_reason'] = f'Insufficient {metric_col}'
            df_result.loc[too_low, 'severity'] = 'High'
        
        # Calculate statistics
        anomaly_count = df_result['threshold_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['threshold_info'] = {
            'min_threshold': rule['min_threshold'],
            'max_threshold': rule['max_threshold'],
            'sustained_minutes': rule['sustained_minutes']
        }
        
        # Store anomaly details
        anomalies = df_result[df_result['threshold_violation']]
        if len(anomalies) > 0:
            self.detected_anomalies.extend(anomalies.to_dict('records'))
        
        return df_result, report


class ResidualAnomalyDetector:
    """
    Model-based anomaly detection using Prophet forecast residuals.
    Detects when actual values deviate significantly from predicted values.
    """
    
    def __init__(self, threshold_std: float = 3.0):
        """
        Initialize residual-based anomaly detector.
        
        Args:
            threshold_std: Number of standard deviations for anomaly threshold
        """
        self.threshold_std = threshold_std
        self.prophet_models = {}
        self.detected_anomalies = []
    
    def detect_anomalies_from_prophet(self, df: pd.DataFrame, forecast_df: pd.DataFrame, 
                                     data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect anomalies by comparing actual vs predicted values.
        
        Args:
            df: Original DataFrame with actual values
            forecast_df: Prophet forecast DataFrame with predictions
            data_type: Type of data
            
        Returns:
            DataFrame with residual-based anomaly flags and report
        """
       
        
        report = {
            'method': 'Prophet Residual-Based',
            'data_type': data_type,
            'threshold_std': self.threshold_std,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        # Identify metric column
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        
        if data_type not in metric_columns:
            st.warning(f"Unknown data type: {data_type}")
            return df, report
        
        metric_col = metric_columns[data_type]
        
        if metric_col not in df.columns:
            st.error(f"Metric column '{metric_col}' not found")
            return df, report
        
        # Merge actual and predicted values
        df_result = df.copy()
        df_result = df_result.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure timestamp is datetime before timezone operations
        df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
        
        # Prophet forecast has columns: ds, yhat, yhat_lower, yhat_upper
        forecast_aligned = forecast_df.copy()
        forecast_aligned = forecast_aligned.rename(columns={'ds': 'timestamp', 'yhat': 'predicted'})
        forecast_aligned['timestamp'] = pd.to_datetime(forecast_aligned['timestamp'])
        
        # Fix timezone mismatch - ensure both timestamp columns are timezone-naive
        if df_result['timestamp'].dt.tz is not None:
            df_result['timestamp'] = df_result['timestamp'].dt.tz_localize(None)
        
        if forecast_aligned['timestamp'].dt.tz is not None:
            forecast_aligned['timestamp'] = forecast_aligned['timestamp'].dt.tz_localize(None)
        
        # Merge on timestamp
        df_result = df_result.merge(
            forecast_aligned[['timestamp', 'predicted', 'yhat_lower', 'yhat_upper']], 
            on='timestamp', 
            how='left'
        )
        
        # Calculate residuals (actual - predicted)
        df_result['residual'] = df_result[metric_col] - df_result['predicted']
        
        # Calculate residual statistics
        residual_mean = df_result['residual'].mean()
        residual_std = df_result['residual'].std()
        
        # Detect anomalies: residuals beyond threshold_std standard deviations
        threshold = self.threshold_std * residual_std
        df_result['residual_anomaly'] = np.abs(df_result['residual']) > threshold
        
        # Also check if value is outside Prophet's confidence interval
        outside_interval = (df_result[metric_col] > df_result['yhat_upper']) | \
                          (df_result[metric_col] < df_result['yhat_lower'])
        
        # Combine both conditions
        df_result['residual_anomaly'] = df_result['residual_anomaly'] | outside_interval
        
        # Add anomaly reason
        df_result['residual_anomaly_reason'] = ''
        df_result.loc[df_result['residual_anomaly'], 'residual_anomaly_reason'] = 'Deviates from predicted trend'
        
        # Calculate statistics
        anomaly_count = df_result['residual_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['residual_stats'] = {
            'mean': float(residual_mean),
            'std': float(residual_std),
            'threshold': float(threshold)
        }
        
        if anomaly_count > 0:
            self.detected_anomalies.extend(
                df_result[df_result['residual_anomaly']].to_dict('records')
            )
        
        return df_result, report


class ClusterAnomalyDetector:
    """
    Cluster-based anomaly detection.
    Identifies data points that are isolated or belong to small/unusual clusters.
    """
    
    def __init__(self):
        self.detected_anomalies = []
        self.cluster_info = {}
    
    def detect_cluster_outliers(self, feature_matrix: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               data_type: str,
                               outlier_threshold: float = 0.05) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect anomalies based on cluster membership.
        Small clusters or isolated points are considered anomalies.
        
        Args:
            feature_matrix: TSFresh feature matrix
            cluster_labels: Cluster assignments from KMeans/DBSCAN
            data_type: Type of data
            outlier_threshold: Clusters smaller than this percentage are anomalies
            
        Returns:
            DataFrame with cluster anomaly flags and report
        """
        
        report = {
            'method': 'Cluster-Based',
            'data_type': data_type,
            'total_clusters': 0,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        df_result = feature_matrix.copy()
        df_result['cluster'] = cluster_labels
        
        # Calculate cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        total_points = len(cluster_labels)
        
        # Identify anomalous clusters (small clusters or noise points)
        anomalous_clusters = []
        
        for cluster_id, size in cluster_sizes.items():
            cluster_percentage = size / total_points
            
            # DBSCAN uses -1 for noise points
            if cluster_id == -1:
                anomalous_clusters.append(cluster_id)
            # Small clusters are anomalies
            elif cluster_percentage < outlier_threshold:
                anomalous_clusters.append(cluster_id)
        
        # Flag anomalies
        df_result['cluster_anomaly'] = df_result['cluster'].isin(anomalous_clusters)
        df_result['cluster_anomaly_reason'] = ''
        
        # Add reasons
        for cluster_id in anomalous_clusters:
            if cluster_id == -1:
                reason = 'Noise point (DBSCAN)'
            else:
                reason = f'Belongs to small cluster #{cluster_id}'
            
            mask = df_result['cluster'] == cluster_id
            df_result.loc[mask, 'cluster_anomaly_reason'] = reason
        
        # Calculate statistics
        anomaly_count = df_result['cluster_anomaly'].sum()
        report['total_clusters'] = int(len(cluster_sizes))
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['cluster_distribution'] = cluster_sizes.to_dict()
        report['anomalous_clusters'] = [int(c) for c in anomalous_clusters]
        
        if anomaly_count > 0:
            self.detected_anomalies.extend(
                df_result[df_result['cluster_anomaly']].to_dict('records')
            )
        
        self.cluster_info = report
        
        return df_result, report


def create_sample_data_with_anomalies() -> Dict[str, pd.DataFrame]:
    """
    Create realistic sample data with intentional anomalies for testing.
    """
    
    # Heart Rate Data with anomalies
    timestamps = pd.date_range(start='2024-01-15 08:00:00', 
                               end='2024-01-15 20:00:00', freq='1min')
    
    base_hr = 70
    hr_data = []
    
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        hr = base_hr
        
        # Normal variations
        if 9 <= time_of_day < 10:  # Morning exercise
            hr = 110 + np.random.normal(0, 5)
        elif 14 <= time_of_day < 15:  # Afternoon activity
            hr = 95 + np.random.normal(0, 5)
        else:
            hr = 70 + np.random.normal(0, 3)
        
        # Inject anomalies
        # Anomaly 1: Sustained high heart rate (tachycardia)
        if 11.5 <= time_of_day < 12:
            hr = 135 + np.random.normal(0, 5)
        
        # Anomaly 2: Low heart rate (bradycardia)
        if 16 <= time_of_day < 16.3:
            hr = 35 + np.random.normal(0, 2)
        
        # Anomaly 3: Sudden spike
        if 18.5 <= time_of_day < 18.6:
            hr = 150
        
        hr_data.append(max(30, min(220, hr)))
    
    heart_rate_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': hr_data
    })
    
    # Step Count Data with anomalies
    step_timestamps = pd.date_range(start='2024-01-15 08:00:00',
                                   end='2024-01-15 20:00:00', freq='5min')
    
    step_data = []
    for i, ts in enumerate(step_timestamps):
        time_of_day = ts.hour + ts.minute / 60
        
        # Normal step patterns
        if 8 <= time_of_day < 9:
            steps = 50 + np.random.randint(-10, 10)
        elif 12 <= time_of_day < 13:
            steps = 80 + np.random.randint(-15, 15)
        elif 17 <= time_of_day < 18:
            steps = 100 + np.random.randint(-20, 20)
        else:
            steps = 20 + np.random.randint(-5, 5)
        
        # Inject anomaly: unrealistic step count
        if 15 <= time_of_day < 15.2:
            steps = 1200  # Impossible in 5 minutes
        
        step_data.append(max(0, steps))
    
    steps_df = pd.DataFrame({
        'timestamp': step_timestamps,
        'step_count': step_data
    })
    
    return {
        'heart_rate': heart_rate_df,
        'steps': steps_df
    }