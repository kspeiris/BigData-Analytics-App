from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

# Custom JSON encoder to handle NumPy and Pandas types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# In-memory storage for active sessions
active_datasets = {}

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj

def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session_data in active_datasets.items():
        session_time = datetime.strptime(session_id[:14], "%Y%m%d%H%M%S")
        if (current_time - session_time).total_seconds() > 3600:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del active_datasets[session_id]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_datasets)
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        cleanup_old_sessions()
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension == 'csv':
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            # Flask uploads are file-like streams; pandas can read them directly.
            # Using pandas for both sizes avoids dask path/startswith errors.
            df = pd.read_csv(file.stream, low_memory=False)
            data = df.to_dict('records')
            columns = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()
        
        elif file_extension == 'json':
            df = pd.read_json(file.stream)
            data = df.to_dict('records')
            columns = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()
        
        else:
            return jsonify({"error": "Unsupported file format. Use CSV or JSON"}), 400

        # Store in memory with session ID
        session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        active_datasets[session_id] = {
            'data': data,
            'columns': columns,
            'dtypes': dtypes,
            'dataframe': df,
            'upload_time': datetime.now().isoformat()
        }

        # Basic statistics
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        basic_stats = {}
        
        for col in numeric_columns:
            basic_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'count': int(df[col].count())
            }

        # Categorical columns analysis
        categorical_columns = [col for col in columns if col not in numeric_columns]
        categorical_stats = {}
        
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_values': int(value_counts.count()),
                'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'top_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            }

        response = {
            "session_id": session_id,
            "message": "File uploaded successfully",
            "summary": convert_numpy_types({
                "rows": len(df),
                "columns": columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "basic_stats": basic_stats,
                "categorical_stats": categorical_stats,
                "file_size": len(str(data))
            })
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/analyze/basic', methods=['POST'])
def basic_analysis():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404
        
        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        
        # Convert all numpy types to native Python types
        description_dict = {}
        for col, stats in df.describe().to_dict().items():
            description_dict[col] = {}
            for stat, val in stats.items():
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    description_dict[col][stat] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    description_dict[col][stat] = float(val)
                else:
                    description_dict[col][stat] = val
        
        analysis_results = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "data_types": df.dtypes.astype(str).to_dict(),
            "description": description_dict,
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }
        
        return jsonify(convert_numpy_types(analysis_results))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/correlation', methods=['POST'])
def correlation_analysis():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404
        
        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return jsonify({"error": "Need at least 2 numeric columns for correlation"}), 400
            
        correlation_matrix = numeric_df.corr().round(4)
        
        # Convert matrix to list with native types
        matrix_list = []
        for row in correlation_matrix.values:
            matrix_list.append([float(x) for x in row])
        
        correlation_data = {
            "columns": correlation_matrix.columns.tolist(),
            "matrix": matrix_list,
            "strong_correlations": []
        }
        
        # Find strong correlations (absolute value > 0.7)
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j and abs(correlation_matrix.iloc[i, j]) > 0.7:
                    correlation_data["strong_correlations"].append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": float(correlation_matrix.iloc[i, j])
                    })
        
        return jsonify(convert_numpy_types(correlation_data))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/clustering', methods=['POST'])
def clustering_analysis():
    try:
        data = request.json
        session_id = data.get('session_id')
        columns = data.get('columns', [])
        n_clusters = data.get('n_clusters', 3)
        
        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404
        
        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        
        # Select specified numeric columns
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        
        if len(columns) < 2:
            return jsonify({"error": "Need at least 2 numeric columns for clustering"}), 400
        
        # Remove rows with missing values
        X = df[columns].dropna().values
        
        if len(X) < n_clusters:
            return jsonify({"error": "Not enough data points for clustering"}), 400
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Convert all numpy arrays to lists with native types
        data_points = []
        for point in X:
            data_points.append([float(x) for x in point])
        
        centers = []
        for center in kmeans.cluster_centers_:
            centers.append([float(x) for x in center])
        
        results = {
            "columns": columns,
            "clusters": [int(x) for x in clusters],
            "centers": centers,
            "data_points": data_points,
            "inertia": float(kmeans.inertia_)
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualize/scatter', methods=['POST'])
def scatter_plot():
    try:
        data = request.json
        session_id = data.get('session_id')
        x_col = data.get('x_column')
        y_col = data.get('y_column')
        
        print(f"Scatter plot request: session_id={session_id}, x_col={x_col}, y_col={y_col}")
        
        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404
        
        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
        # Check if columns exist
        if x_col not in df.columns:
            return jsonify({"error": f"X column '{x_col}' not found in dataset"}), 400
        if y_col not in df.columns:
            return jsonify({"error": f"Y column '{y_col}' not found in dataset"}), 400
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[x_col]):
            return jsonify({"error": f"X column '{x_col}' is not numeric"}), 400
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return jsonify({"error": f"Y column '{y_col}' is not numeric"}), 400
        
        # Remove rows with missing values in these columns
        plot_data = df[[x_col, y_col]].dropna()
        
        if len(plot_data) == 0:
            return jsonify({"error": "No valid data points after removing missing values"}), 400
        
        print(f"Plotting {len(plot_data)} data points")
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, s=50)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.title(f'{y_col} vs {x_col}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save plot to bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()
        
        # Encode image to base64
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        return jsonify({
            "image": f"data:image/png;base64,{img_base64}",
            "message": f"Scatter plot created with {len(plot_data)} points"
        })
    
    except Exception as e:
        print(f"Scatter plot error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/analyze/trend', methods=['POST'])
def trend_analysis():
    try:
        data = request.json
        session_id = data.get('session_id')
        column = data.get('column')
        
        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404
        
        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        
        if column not in df.columns:
            return jsonify({"error": "Column not found"}), 400
        
        # Simple trend analysis
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if len(series) < 2:
            return jsonify({"error": "Not enough data points for trend analysis"}), 400
        
        # Calculate trend line
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series.values, 1)
        trend_line = slope * x + intercept
        
        results = {
            "data_points": [float(x) for x in series.values],
            "trend_line": [float(x) for x in trend_line],
            "slope": float(slope),
            "intercept": float(intercept),
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "correlation": float(np.corrcoef(x, series.values)[0, 1])
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/outliers', methods=['POST'])
def outlier_analysis():
    try:
        data = request.json
        session_id = data.get('session_id')
        z_threshold = float(data.get('z_threshold', 3.0))

        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404

        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if numeric_df.shape[1] < 1:
            return jsonify({"error": "Need at least 1 numeric column for outlier analysis"}), 400

        z_outlier_mask = pd.DataFrame(False, index=numeric_df.index, columns=numeric_df.columns)
        outlier_summary = []

        for col in numeric_df.columns:
            series = pd.to_numeric(numeric_df[col], errors='coerce')
            mean = series.mean()
            std = series.std()
            if pd.isna(std) or std == 0:
                outlier_count = 0
            else:
                z_scores = ((series - mean).abs() / std)
                col_mask = z_scores > z_threshold
                z_outlier_mask[col] = col_mask.fillna(False)
                outlier_count = int(col_mask.sum())

            outlier_summary.append({
                "column": col,
                "outlier_count": outlier_count,
                "outlier_percent": float((outlier_count / len(series)) * 100) if len(series) else 0.0
            })

        row_outlier_mask = z_outlier_mask.any(axis=1)
        outlier_rows = df[row_outlier_mask].head(50).to_dict('records')

        # IsolationForest gives a stronger row-level outlier signal across all numeric columns.
        iso_rows = numeric_df.dropna()
        isolation_summary = {"rows_scored": 0, "anomalies": 0, "anomaly_percent": 0.0}
        if len(iso_rows) >= 10:
            contamination = min(0.1, max(0.01, 10 / len(iso_rows)))
            model = IsolationForest(random_state=42, contamination=contamination)
            preds = model.fit_predict(iso_rows)
            anomaly_count = int((preds == -1).sum())
            isolation_summary = {
                "rows_scored": int(len(iso_rows)),
                "anomalies": anomaly_count,
                "anomaly_percent": float((anomaly_count / len(iso_rows)) * 100)
            }

        results = {
            "z_threshold": z_threshold,
            "total_rows": int(len(df)),
            "rows_with_any_outlier": int(row_outlier_mask.sum()),
            "rows_with_any_outlier_percent": float((row_outlier_mask.sum() / len(df)) * 100) if len(df) else 0.0,
            "column_summary": sorted(outlier_summary, key=lambda x: x["outlier_count"], reverse=True),
            "sample_outlier_rows": outlier_rows,
            "isolation_forest": isolation_summary
        }

        return jsonify(convert_numpy_types(results))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/pca', methods=['POST'])
def pca_analysis():
    try:
        data = request.json
        session_id = data.get('session_id')
        n_components = int(data.get('n_components', 2))

        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404

        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if numeric_df.shape[1] < 2:
            return jsonify({"error": "Need at least 2 numeric columns for PCA"}), 400

        clean_df = numeric_df.dropna()
        if clean_df.shape[0] < 2:
            return jsonify({"error": "Not enough valid rows for PCA after removing missing values"}), 400

        n_components = max(2, min(n_components, clean_df.shape[1]))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clean_df.values)

        pca = PCA(n_components=n_components, random_state=42)
        components = pca.fit_transform(scaled)

        plot_points = []
        for i, point in enumerate(components[:300]):
            plot_points.append({
                "pc1": float(point[0]) if len(point) > 0 else 0.0,
                "pc2": float(point[1]) if len(point) > 1 else 0.0,
                "index": int(i)
            })

        loadings = []
        for idx, col in enumerate(clean_df.columns):
            loadings.append({
                "column": col,
                "pc1_loading": float(pca.components_[0][idx]) if pca.components_.shape[0] > 0 else 0.0,
                "pc2_loading": float(pca.components_[1][idx]) if pca.components_.shape[0] > 1 else 0.0
            })

        results = {
            "columns_used": clean_df.columns.tolist(),
            "rows_used": int(clean_df.shape[0]),
            "n_components": int(n_components),
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "points": plot_points,
            "feature_loadings": loadings
        }

        return jsonify(convert_numpy_types(results))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/dataset', methods=['POST'])
def debug_dataset():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404
        
        dataset = active_datasets[session_id]
        df = pd.DataFrame(dataset['data'])
        
        debug_info = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "shape": df.shape,
            "sample_data": df.head(3).to_dict('records')
        }
        
        return jsonify(debug_info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/session/clear', methods=['POST'])
def clear_session():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id in active_datasets:
            del active_datasets[session_id]
            return jsonify({"message": "Session cleared successfully"})
        else:
            return jsonify({"error": "Session not found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
