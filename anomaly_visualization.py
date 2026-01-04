"""
Anomaly Visualization Module for FitPulse
Contains all visualization components for anomaly detection results
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class AnomalyVisualizer:
    """
    Creates interactive visualizations highlighting detected anomalies.
    """
    
    def __init__(self):
        self.color_scheme = {
            'normal': '#1f77b4',  # Blue
            'threshold': '#ff7f0e',  # Orange
            'residual': '#d62728',  # Red
            'cluster': '#9467bd',  # Purple
            'combined': '#e377c2'  # Pink
        }
    
    def plot_heart_rate_anomalies(self, df: pd.DataFrame, title: str = "Heart Rate Anomaly Detection"):
        """Plot heart rate data with all detected anomalies highlighted."""
        
        fig = go.Figure()
        
        # Check which anomaly columns exist
        has_threshold = 'threshold_anomaly' in df.columns
        has_threshold_violation = 'threshold_violation' in df.columns
        has_residual = 'residual_anomaly' in df.columns
        has_cluster = 'cluster_anomaly' in df.columns
        has_predicted = 'predicted' in df.columns
        
        # Plot normal data (baseline) - exclude all detected anomalies
        normal_data = df.copy()
        anomaly_mask = False
        if has_threshold:
            anomaly_mask = anomaly_mask | normal_data['threshold_anomaly']
        if has_residual:
            anomaly_mask = anomaly_mask | normal_data['residual_anomaly']
        if has_cluster:
            anomaly_mask = anomaly_mask | normal_data['cluster_anomaly']
        
        if anomaly_mask.any():
            normal_data = normal_data[~anomaly_mask]
        
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['heart_rate'],
            mode='lines',
            name='Normal Heart Rate',
            line=dict(color=self.color_scheme['normal'], width=1.5),
            hovertemplate='<b>Time:</b> %{x}<br><b>HR:</b> %{y} bpm<extra></extra>'
        ))
        
        # Plot predicted values if available
        if has_predicted:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['predicted'],
                mode='lines',
                name='Predicted Trend (Prophet)',
                line=dict(color='lightgreen', width=2, dash='dash'),
                opacity=0.7,
                hovertemplate='<b>Time:</b> %{x}<br><b>Predicted:</b> %{y:.1f} bpm<extra></extra>'
            ))
            
            # Add confidence interval
            if 'yhat_upper' in df.columns and 'yhat_lower' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['yhat_upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['yhat_lower'],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(144, 238, 144, 0.2)',
                    hoverinfo='skip'
                ))
        
        # Plot threshold anomalies
        # 🔸 Plot ALL threshold violations (visual only)
        if has_threshold_violation:
            violations = df[df['threshold_violation']]
            if len(violations) > 0:
                fig.add_trace(go.Scatter(
                    x=violations['timestamp'],
                    y=violations['heart_rate'],
                    mode='markers',
                    name='Threshold Violation (Instant)',
                    marker=dict(
                        color='orange',
                        size=6
                    ),
                    hovertemplate='<b>⚠️ Threshold Violation</b><br>Time: %{x}<br>HR: %{y} bpm<extra></extra>'
                ))

        # ❌ Plot sustained threshold anomalies (confirmed)
        if has_threshold:
            sustained = df[df['threshold_anomaly']]
            if len(sustained) > 0:
                fig.add_trace(go.Scatter(
                    x=sustained['timestamp'],
                    y=sustained['heart_rate'],
                    mode='markers',
                    name='Threshold Anomaly (Sustained)',
                    marker=dict(
                        color=self.color_scheme['threshold'],
                        size=11,
                        symbol='x',
                        line=dict(width=2)
                    ),
                    hovertemplate='<b>❌ Sustained Threshold Anomaly</b><br>Time: %{x}<br>HR: %{y} bpm<extra></extra>'
                ))

        
        # Plot residual anomalies
        if has_residual:
            residual_anomalies = df[df['residual_anomaly']]
            if len(residual_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=residual_anomalies['timestamp'],
                    y=residual_anomalies['heart_rate'],
                    mode='markers',
                    name='Model Deviation Anomalies',
                    marker=dict(
                        color=self.color_scheme['residual'],
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>🔴 Model Deviation</b><br>Time: %{x}<br>HR: %{y} bpm<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Heart Rate (bpm)",
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(240,240,240,0.5)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='white'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_steps_anomalies(self, df: pd.DataFrame, title: str = "Step Count Anomaly Detection"):
        """Plot step count data with anomalies."""
        
        fig = go.Figure()
        
        has_threshold = 'threshold_anomaly' in df.columns
        has_residual = 'residual_anomaly' in df.columns
        has_predicted = 'predicted' in df.columns
        
        # Normal data - exclude all anomalies
        normal_data = df.copy()
        anomaly_mask = False
        if has_threshold:
            anomaly_mask = anomaly_mask | normal_data['threshold_anomaly']
        if has_residual:
            anomaly_mask = anomaly_mask | normal_data['residual_anomaly']
        
        if anomaly_mask.any():
            normal_data = normal_data[~anomaly_mask]
        
        fig.add_trace(go.Bar(
            x=normal_data['timestamp'],
            y=normal_data['step_count'],
            name='Normal Step Count',
            marker_color=self.color_scheme['normal'],
            opacity=0.7,
            hovertemplate='<b>Time:</b> %{x}<br><b>Steps:</b> %{y}<extra></extra>'
        ))
        
        # Predicted trend
        if has_predicted:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['predicted'],
                mode='lines',
                name='Predicted Trend',
                line=dict(color='orange', width=3, dash='dot'),
                hovertemplate='<b>Predicted:</b> %{y:.0f} steps<extra></extra>'
            ))
        
        # Threshold anomalies
        if has_threshold:
            threshold_anomalies = df[df['threshold_anomaly']]
            if len(threshold_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=threshold_anomalies['timestamp'],
                    y=threshold_anomalies['step_count'],
                    mode='markers',
                    name='Threshold Anomalies',
                    marker=dict(
                        color=self.color_scheme['threshold'],
                        size=15,
                        symbol='x',
                        line=dict(width=2)
                    ),
                    hovertemplate='<b>⚠️ Threshold Anomaly</b><br>Time: %{x}<br>Steps: %{y}<extra></extra>'
                ))
        
        # Residual anomalies
        if has_residual:
            residual_anomalies = df[df['residual_anomaly']]
            if len(residual_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=residual_anomalies['timestamp'],
                    y=residual_anomalies['step_count'],
                    mode='markers',
                    name='Model Deviation Anomalies',
                    marker=dict(
                        color=self.color_scheme['residual'],
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>🔴 Model Deviation</b><br>Time: %{x}<br>Steps: %{y}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Step Count",
            hovermode='x unified',
            height=500,
            showlegend=True,
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_sleep_anomalies(self, df: pd.DataFrame, title: str = "Sleep Pattern Anomaly Detection"):
        """Plot sleep duration data with anomalies."""
        
        fig = go.Figure()
        
        has_threshold = 'threshold_anomaly' in df.columns
        has_residual = 'residual_anomaly' in df.columns
        
        # Convert minutes to hours for better readability
        df['duration_hours'] = df['duration_minutes'] / 60
        
        # Normal sleep - exclude all anomalies
        normal_data = df.copy()
        anomaly_mask = False
        if has_threshold:
            anomaly_mask = anomaly_mask | normal_data['threshold_anomaly']
        if has_residual:
            anomaly_mask = anomaly_mask | normal_data['residual_anomaly']
        
        if anomaly_mask.any():
            normal_data = normal_data[~anomaly_mask]
        
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['duration_hours'],
            mode='lines+markers',
            name='Normal Sleep Duration',
            line=dict(color=self.color_scheme['normal'], width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Sleep:</b> %{y:.1f} hours<extra></extra>'
        ))
        
        # Add reference lines
        fig.add_hline(y=7, line_dash="dash", line_color="green", 
                     annotation_text="Recommended (7h)", annotation_position="right")
        fig.add_hline(y=3, line_dash="dash", line_color="red", 
                     annotation_text="Minimum (3h)", annotation_position="right")
        fig.add_hline(y=12, line_dash="dash", line_color="red", 
                     annotation_text="Maximum (12h)", annotation_position="right")
        
        # Threshold anomalies
        if has_threshold:
            threshold_anomalies = df[df['threshold_anomaly']]
            if len(threshold_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=threshold_anomalies['timestamp'],
                    y=threshold_anomalies['duration_hours'],
                    mode='markers',
                    name='Threshold Anomalies',
                    marker=dict(
                        color=self.color_scheme['threshold'],
                        size=15,
                        symbol='x',
                        line=dict(width=2)
                    ),
                    hovertemplate='<b>⚠️ Threshold Anomaly</b><br>Date: %{x}<br>Duration: %{y:.1f} hours<extra></extra>'
                ))
        
        # Residual anomalies
        if has_residual:
            residual_anomalies = df[df['residual_anomaly']]
            if len(residual_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=residual_anomalies['timestamp'],
                    y=residual_anomalies['duration_hours'],
                    mode='markers',
                    name='Model Deviation Anomalies',
                    marker=dict(
                        color=self.color_scheme['residual'],
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>🔴 Model Deviation</b><br>Date: %{x}<br>Duration: %{y:.1f} hours<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sleep Duration (hours)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            yaxis=dict(range=[0, 14])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_anomaly_summary_dashboard(self, all_reports: Dict):
        """Create a comprehensive dashboard showing all anomaly detection results."""
        
        st.subheader("📊 Anomaly Detection Summary Dashboard")
        
        # Collect all anomaly counts
        total_anomalies = 0
        detection_methods = []
        
        for data_type, reports in all_reports.items():
            for method, report in reports.items():
                if 'anomalies_detected' in report:
                    total_anomalies += report['anomalies_detected']
                    detection_methods.append({
                        'Data Type': data_type.replace('_', ' ').title(),
                        'Method': report.get('method', method),
                        'Anomalies': report['anomalies_detected'],
                        'Percentage': f"{report.get('anomaly_percentage', 0):.2f}%"
                    })
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies Detected", total_anomalies)
        
        with col2:
            st.metric("Detection Methods Used", len(set([d['Method'] for d in detection_methods])))
        
        with col3:
            data_types = len(set([d['Data Type'] for d in detection_methods]))
            st.metric("Data Types Analyzed", data_types)
        
        # Detailed table
        if detection_methods:
            st.subheader("Anomaly Detection Breakdown")
            summary_df = pd.DataFrame(detection_methods)
            st.dataframe(summary_df, use_container_width=True)
            
            # Create comparison bar chart
            fig = go.Figure()
            
            for method in summary_df['Method'].unique():
                method_data = summary_df[summary_df['Method'] == method]
                fig.add_trace(go.Bar(
                    name=method,
                    x=method_data['Data Type'],
                    y=method_data['Anomalies'],
                    text=method_data['Anomalies'],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Anomalies Detected by Method and Data Type",
                xaxis_title="Data Type",
                yaxis_title="Number of Anomalies",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_anomaly_timeline(self, all_data: Dict):
        """Create a timeline view of all anomalies across data types."""
        
        st.subheader("📅 Anomaly Timeline")
        
        fig = go.Figure()
        
        colors = ['red', 'orange', 'purple', 'brown']
        
        for i, (data_type, df) in enumerate(all_data.items()):
            # Find all anomaly columns
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and df[col].dtype == bool]
            
            if anomaly_cols:
                # Get all anomalies for this data type
                anomaly_mask = df[anomaly_cols].any(axis=1)
                anomalies = df[anomaly_mask]
                
                if len(anomalies) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies['timestamp'],
                        y=[data_type] * len(anomalies),
                        mode='markers',
                        name=f'{data_type.replace("_", " ").title()} Anomalies',
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=12,
                            symbol='diamond'
                        ),
                        hovertemplate=f'<b>{data_type.title()} Anomaly</b><br>Time: %{{x}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title="Anomaly Timeline Across All Data Types",
            xaxis_title="Time",
            yaxis_title="Data Type",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def export_anomaly_report(self, results: Dict):
        """Create export functionality for anomaly reports."""
        
        st.subheader("💾 Export Anomaly Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare JSON report data
            report_json = json.dumps(results['reports'], indent=2, default=str)
            st.download_button(
                label="📄 Export Anomaly Report (JSON)",
                data=report_json,
                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_json_viz",
                use_container_width=True
            )
        
        with col2:
            # Prepare CSV data
            all_anomalies = []
            for data_type, df in results['data_with_anomalies'].items():
                # Get rows with any anomaly
                anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and df[col].dtype == bool]
                if anomaly_cols:
                    anomaly_mask = df[anomaly_cols].any(axis=1)
                    anomalies = df[anomaly_mask].copy()
                    anomalies['data_type'] = data_type
                    all_anomalies.append(anomalies)
            
            if all_anomalies:
                combined_df = pd.concat(all_anomalies, ignore_index=True)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="📊 Export Anomaly Data (CSV)",
                    data=csv,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv_viz",
                    use_container_width=True
                )
            else:
                st.button(
                    "📊 No Anomaly Data Available",
                    disabled=True,
                    use_container_width=True
                )
    
    def create_severity_analysis(self, all_data: Dict):
        """Create severity analysis of detected anomalies."""
        
        st.subheader("⚠️ Anomaly Severity Analysis")
        
        severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for data_type, df in all_data.items():
            if 'severity' in df.columns:
                severity_data = df['severity'].value_counts()
                for severity, count in severity_data.items():
                    if severity in severity_counts:
                        severity_counts[severity] += count
        
        if sum(severity_counts.values()) > 0:
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                hole=0.3,
                marker_colors=['#ff4444', '#ff8800', '#ffcc00']
            )])
            
            fig.update_layout(
                title="Anomaly Severity Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display severity metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔴 High Severity", severity_counts['High'])
            with col2:
                st.metric("🟡 Medium Severity", severity_counts['Medium'])
            with col3:
                st.metric("🟢 Low Severity", severity_counts['Low'])
        else:
            st.info("No severity data available for analysis.")