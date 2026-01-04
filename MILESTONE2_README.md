# 🚀 FitPulse - Milestone 2: Advanced ML Features

## 📋 Overview

Milestone 2 extends FitPulse with advanced machine learning capabilities including time-series feature extraction, forecasting, and behavioral pattern analysis. Built on top of Milestone 1's preprocessed data.

## ✨ Features

### 🔵 TSFresh Feature Extraction
- **Statistical Features**: 19+ time-series features per window
- **Rolling Windows**: Configurable window sizes (10-120 minutes)
- **Feature Types**: Mean, median, variance, skewness, kurtosis, trends
- **Automatic Cleaning**: Removes constant and low-variance features

### 🟡 Prophet Forecasting
- **Time Series Modeling**: Facebook Prophet for trend analysis
- **Confidence Intervals**: 95% prediction intervals
- **Seasonality Detection**: Daily, weekly, yearly patterns
- **Anomaly Detection**: Residual-based outlier identification
- **Component Analysis**: Trend and seasonality decomposition

### 🟢 Behavioral Clustering
- **Pattern Recognition**: KMeans and DBSCAN clustering
- **Dimensionality Reduction**: PCA and t-SNE visualization
- **Cluster Analysis**: Silhouette and Davies-Bouldin scoring
- **Interactive Visualization**: 2D cluster plots with statistics

## 🚀 Quick Start

### Prerequisites
```bash
# Install additional ML dependencies
pip install tsfresh prophet scikit-learn
```

### Usage Flow
1. **Complete Milestone 1**: Preprocess your data first
2. **Configure ML Settings**: Set window size, forecast periods, clustering method
3. **Run ML Pipeline**: Click "🚀 Run Milestone 2 Pipeline"
4. **Explore Results**: Review features, forecasts, and clusters

## 🔬 ML Pipeline Steps

### Step 1: Feature Extraction
```
Input: Preprocessed time series data
Process: TSFresh statistical feature extraction
Output: Feature matrix (windows × features)
```

**Features Extracted:**
- **Central Tendency**: Mean, median, mode
- **Dispersion**: Standard deviation, variance, range
- **Shape**: Skewness, kurtosis
- **Trends**: Linear trend slopes
- **Energy**: Absolute energy, sum of changes
- **Autocorrelation**: Lag-1 and lag-2 correlations

### Step 2: Trend Modeling
```
Input: Original time series data
Process: Prophet model training and forecasting
Output: Predictions with confidence intervals
```

**Prophet Features:**
- **Trend Analysis**: Long-term direction identification
- **Seasonality**: Daily/weekly pattern detection
- **Forecasting**: Future value predictions
- **Uncertainty**: Confidence interval estimation
- **Anomaly Detection**: Residual-based outliers

### Step 3: Clustering Analysis
```
Input: Extracted feature matrix
Process: Behavioral pattern clustering
Output: Cluster assignments and visualizations
```

**Clustering Methods:**
- **KMeans**: Partition-based clustering (2-10 clusters)
- **DBSCAN**: Density-based clustering with noise detection
- **Evaluation**: Silhouette score and Davies-Bouldin index
- **Visualization**: PCA/t-SNE 2D projections

## ⚙️ Configuration Options

### Feature Extraction Settings
- **Window Size**: 10-120 minutes (default: 60)
- **Overlap**: 50% window overlap for more samples
- **Complexity**: Minimal, efficient, or comprehensive feature sets

### Forecasting Settings
- **Forecast Periods**: 50-500 future points (default: 100)
- **Seasonality**: Automatic daily pattern detection
- **Confidence Level**: 95% prediction intervals

### Clustering Settings
- **Method**: KMeans or DBSCAN
- **Clusters**: 2-10 for KMeans (default: 3)
- **Visualization**: PCA or t-SNE dimensionality reduction

## 📊 Output & Results

### Feature Matrix
- **Dimensions**: Windows × Features
- **Feature Names**: Descriptive statistical feature names
- **Quality**: Cleaned and normalized features
- **Visualization**: Top features by variance

### Forecast Results
- **Predictions**: Future value forecasts
- **Confidence Bands**: Upper and lower bounds
- **Components**: Trend and seasonality breakdown
- **Residuals**: Model error analysis
- **Anomalies**: Outlier detection results

### Cluster Analysis
- **Assignments**: Cluster labels for each window
- **Statistics**: Cluster sizes and characteristics
- **Visualization**: 2D scatter plots with cluster colors
- **Metrics**: Clustering quality scores

## 🎯 Use Cases

### Health Monitoring
- **Pattern Detection**: Identify daily activity patterns
- **Anomaly Detection**: Spot unusual health metrics
- **Trend Analysis**: Track long-term health trends
- **Forecasting**: Predict future health metrics

### Fitness Analytics
- **Workout Patterns**: Cluster exercise behaviors
- **Performance Trends**: Analyze fitness improvements
- **Recovery Analysis**: Identify rest and recovery patterns
- **Goal Tracking**: Forecast progress toward fitness goals

### Sleep Analysis
- **Sleep Patterns**: Identify sleep behavior clusters
- **Quality Trends**: Track sleep quality over time
- **Anomaly Detection**: Spot sleep disturbances
- **Circadian Analysis**: Understand daily sleep cycles

## 📈 Performance Metrics

### Feature Extraction
- **Speed**: ~0.1-0.5 seconds per dataset
- **Features**: 19+ statistical features per window
- **Windows**: Depends on data length and window size
- **Memory**: Efficient processing for large datasets

### Forecasting
- **Training Time**: 1-3 seconds per model
- **Accuracy**: MAE, RMSE, MAPE metrics provided
- **Forecast Horizon**: Configurable future periods
- **Confidence**: 95% prediction intervals

### Clustering
- **Speed**: 0.1-1 second per analysis
- **Quality**: Silhouette score > 0.5 target
- **Visualization**: Real-time 2D projections
- **Scalability**: Handles 100+ feature dimensions

## 🔧 Technical Implementation

### Core Classes
- `TSFreshFeatureExtractor`: Time-series feature extraction
- `ProphetTrendModeler`: Forecasting and trend analysis
- `BehaviorClusterer`: Pattern clustering and visualization
- `FeatureModelingPipeline`: ML pipeline orchestrator

### Key Dependencies
- `tsfresh`: Time-series feature extraction
- `prophet`: Forecasting and trend modeling
- `scikit-learn`: Clustering and dimensionality reduction
- `plotly`: Interactive ML visualizations

### Data Flow
```
Preprocessed Data → Feature Extraction → Clustering
                 ↓
              Forecasting → Anomaly Detection
```

## 🛠️ Troubleshooting

### Common Issues
- **Empty Features**: Check data has sufficient rows for window size
- **Prophet Errors**: Ensure timestamps are timezone-naive
- **Clustering Fails**: Verify feature matrix is not empty
- **Memory Issues**: Reduce window size or use minimal features

### Debug Features
- **Data Inspection**: Shows input data shape and columns
- **Processing Logs**: Real-time status updates
- **Error Messages**: Detailed error descriptions
- **Quality Metrics**: Feature and model quality scores

## 📞 Advanced Configuration

### Custom Feature Sets
```python
# Efficient feature set (default)
features = {
    "mean", "median", "std", "variance",
    "minimum", "maximum", "skewness", "kurtosis",
    "quantile", "linear_trend", "autocorrelation"
}
```

### Prophet Parameters
```python
# Model configuration
prophet_config = {
    "daily_seasonality": True,
    "weekly_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "interval_width": 0.95
}
```

### Clustering Options
```python
# KMeans configuration
kmeans_config = {
    "n_clusters": 3,
    "random_state": 42,
    "n_init": 10
}

# DBSCAN configuration
dbscan_config = {
    "eps": 0.5,
    "min_samples": 5
}
```

---

**Next Steps:** Use the extracted features and insights for personalized health recommendations, automated alerts, or integration with other health platforms.