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
import warnings
from dotenv import load_dotenv

load_dotenv()

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

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

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

        session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        active_datasets[session_id] = {
            'data': data,
            'columns': columns,
            'dtypes': dtypes,
            'dataframe': df,
            'upload_time': datetime.now().isoformat()
        }

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
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return jsonify({"error": "Need at least 2 numeric columns for correlation"}), 400
            
        correlation_matrix = numeric_df.corr().round(4)
        
        matrix_list = []
        for row in correlation_matrix.values:
            matrix_list.append([float(x) for x in row])
        
        correlation_data = {
            "columns": correlation_matrix.columns.tolist(),
            "matrix": matrix_list,
            "strong_correlations": []
        }
        
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
        
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        
        if len(columns) < 2:
            return jsonify({"error": "Need at least 2 numeric columns for clustering"}), 400
        
        X = df[columns].dropna().values
        
        if len(X) < n_clusters:
            return jsonify({"error": "Not enough data points for clustering"}), 400
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
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
        
        if x_col not in df.columns:
            return jsonify({"error": f"X column '{x_col}' not found in dataset"}), 400
        if y_col not in df.columns:
            return jsonify({"error": f"Y column '{y_col}' not found in dataset"}), 400
        
        if not pd.api.types.is_numeric_dtype(df[x_col]):
            return jsonify({"error": f"X column '{x_col}' is not numeric"}), 400
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return jsonify({"error": f"Y column '{y_col}' is not numeric"}), 400
        
        plot_data = df[[x_col, y_col]].dropna()
        
        if len(plot_data) == 0:
            return jsonify({"error": "No valid data points after removing missing values"}), 400
        
        print(f"Plotting {len(plot_data)} data points")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, s=50)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.title(f'{y_col} vs {x_col}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close()
        
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
        
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        
        if len(series) < 2:
            return jsonify({"error": "Not enough data points for trend analysis"}), 400
        
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

@app.route('/analyze/charts', methods=['POST'])
def charts_analysis():
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        filters = data.get('filters', {})

        if session_id not in active_datasets:
            return jsonify({"error": "Session not found"}), 404

        dataset = active_datasets[session_id]
        full_df = pd.DataFrame(dataset['data'])

        if full_df.empty:
            return jsonify({"error": "Dataset is empty"}), 400

        def to_float_or_none(value):
            if value is None or value == "":
                return None
            try:
                return float(value)
            except Exception:
                return None

        all_columns = full_df.columns.tolist()
        all_numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
        all_categorical_cols = [c for c in all_columns if c not in all_numeric_cols]

        numeric_filter_meta = []
        for col in all_numeric_cols:
            series = pd.to_numeric(full_df[col], errors='coerce').dropna()
            if len(series) == 0:
                continue
            numeric_filter_meta.append({
                "column": col,
                "min": float(series.min()),
                "max": float(series.max())
            })

        categorical_filter_meta = []
        for col in all_categorical_cols:
            values = (
                full_df[col]
                .astype(str)
                .replace("nan", np.nan)
                .dropna()
                .value_counts()
                .head(30)
                .index
                .tolist()
            )
            if values:
                categorical_filter_meta.append({"column": col, "values": [str(v) for v in values]})

        date_filter_meta = []
        for col in all_columns:
            if col in all_numeric_cols:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                parsed = pd.to_datetime(full_df[col], errors='coerce', utc=True)
            valid = parsed.dropna()
            if len(valid) >= max(5, int(len(full_df) * 0.4)):
                date_filter_meta.append({
                    "column": col,
                    "min": valid.min().date().isoformat(),
                    "max": valid.max().date().isoformat()
                })

        df = full_df.copy()

        range_filter = filters.get("range", {}) if isinstance(filters, dict) else {}
        range_col = range_filter.get("column")
        if range_col in df.columns:
            series = pd.to_numeric(df[range_col], errors='coerce')
            min_val = to_float_or_none(range_filter.get("min"))
            max_val = to_float_or_none(range_filter.get("max"))
            if min_val is not None:
                df = df[series >= min_val]
            if max_val is not None:
                df = df[pd.to_numeric(df[range_col], errors='coerce') <= max_val]

        category_filter = filters.get("category", {}) if isinstance(filters, dict) else {}
        cat_col = category_filter.get("column")
        cat_val = category_filter.get("value")
        if cat_col in df.columns and cat_val not in [None, "", "ALL"]:
            df = df[df[cat_col].astype(str) == str(cat_val)]

        date_filter = filters.get("date", {}) if isinstance(filters, dict) else {}
        date_col = date_filter.get("column")
        if date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors='coerce', utc=True)
            start = date_filter.get("start")
            end = date_filter.get("end")
            if start:
                start_dt = pd.to_datetime(start, errors='coerce', utc=True)
                if pd.notna(start_dt):
                    df = df[parsed >= start_dt]
                    parsed = pd.to_datetime(df[date_col], errors='coerce', utc=True)
            if end:
                end_dt = pd.to_datetime(end, errors='coerce', utc=True)
                if pd.notna(end_dt):
                    df = df[parsed <= end_dt]

        if df.empty:
            return jsonify({
                "summary": {
                    "rows": 0,
                    "columns": int(len(all_columns)),
                    "numeric_columns": int(len(all_numeric_cols)),
                    "categorical_columns": int(len(all_categorical_cols)),
                    "total_missing_cells": 0,
                    "rows_before_filter": int(len(full_df)),
                    "rows_after_filter": 0
                },
                "missing_values": [],
                "distributions": [],
                "boxplot_stats": [],
                "correlation": {"columns": [], "matrix": []},
                "scatter_pairs": [],
                "categorical_breakdown": [],
                "trend_series": [],
                "pca_variance": [],
                "filter_metadata": {
                    "numeric_columns": numeric_filter_meta,
                    "categorical_columns": categorical_filter_meta,
                    "date_columns": date_filter_meta
                }
            })

        rows = int(len(df))
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in columns if c not in numeric_cols]

        missing_values = []
        for col in columns:
            missing = int(df[col].isna().sum())
            missing_values.append({
                "column": col,
                "missing": missing,
                "missing_percent": float((missing / rows) * 100) if rows else 0.0
            })
        missing_values = sorted(missing_values, key=lambda x: x["missing"], reverse=True)

        distributions = []
        for col in numeric_cols[:6]:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) < 2:
                continue
            counts, edges = np.histogram(series.values, bins=12)
            bins = []
            for i in range(len(counts)):
                bins.append({
                    "start": float(edges[i]),
                    "end": float(edges[i + 1]),
                    "count": int(counts[i]),
                    "label": f"{edges[i]:.2f} - {edges[i + 1]:.2f}"
                })
            distributions.append({
                "column": col,
                "bins": bins
            })

        boxplot_stats = []
        for col in numeric_cols[:10]:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) < 2:
                continue
            boxplot_stats.append({
                "column": col,
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0
            })

        correlation = {"columns": [], "matrix": []}
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols[:10]].corr().round(4)
            correlation = {
                "columns": corr_df.columns.tolist(),
                "matrix": [[float(x) for x in row] for row in corr_df.values]
            }

        scatter_pairs = []
        if len(numeric_cols) >= 2:
            candidate_cols = numeric_cols[:5]
            pair_count = 0
            for i in range(len(candidate_cols)):
                for j in range(i + 1, len(candidate_cols)):
                    if pair_count >= 4:
                        break
                    x_col = candidate_cols[i]
                    y_col = candidate_cols[j]
                    pair_df = df[[x_col, y_col]].dropna().head(500)
                    points = []
                    for _, r in pair_df.iterrows():
                        points.append({"x": float(r[x_col]), "y": float(r[y_col])})
                    scatter_pairs.append({
                        "x_column": x_col,
                        "y_column": y_col,
                        "points": points
                    })
                    pair_count += 1
                if pair_count >= 4:
                    break

        categorical_breakdown = []
        for col in categorical_cols[:6]:
            value_counts = df[col].astype(str).value_counts().head(8)
            values = []
            for idx, val in value_counts.items():
                values.append({"category": str(idx), "count": int(val)})
            categorical_breakdown.append({
                "column": col,
                "values": values
            })

        trend_series = []
        for col in numeric_cols[:3]:
            series = pd.to_numeric(df[col], errors='coerce')
            valid = series.dropna()
            if len(valid) < 2:
                continue
            sampled = valid.head(300)
            points = [{"index": int(i), "value": float(v)} for i, v in enumerate(sampled.values)]
            trend_series.append({"column": col, "points": points})

        pca_variance = []
        if len(numeric_cols) >= 2:
            pca_df = df[numeric_cols].dropna()
            if len(pca_df) >= 2:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(pca_df.values)
                comps = min(5, pca_df.shape[1])
                pca = PCA(n_components=comps, random_state=42)
                pca.fit(scaled)
                for i, v in enumerate(pca.explained_variance_ratio_):
                    pca_variance.append({
                        "component": f"PC{i + 1}",
                        "variance": float(v),
                        "variance_percent": float(v * 100)
                    })

        payload = {
            "summary": {
                "rows": rows,
                "columns": int(len(columns)),
                "numeric_columns": int(len(numeric_cols)),
                "categorical_columns": int(len(categorical_cols)),
                "total_missing_cells": int(df.isna().sum().sum()),
                "rows_before_filter": int(len(full_df)),
                "rows_after_filter": rows
            },
            "missing_values": missing_values,
            "distributions": distributions,
            "boxplot_stats": boxplot_stats,
            "correlation": correlation,
            "scatter_pairs": scatter_pairs,
            "categorical_breakdown": categorical_breakdown,
            "trend_series": trend_series,
            "pca_variance": pca_variance,
            "filter_metadata": {
                "numeric_columns": numeric_filter_meta,
                "categorical_columns": categorical_filter_meta,
                "date_columns": date_filter_meta
            }
        }

        return jsonify(convert_numpy_types(payload))

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

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
