import React from 'react';

const DataSummary = ({ analysis }) => {
  if (!analysis) {
    return (
      <div className="container">
        <div className="card">
          <div className="empty-state">
            <h3>No data available</h3>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="card">
        <div className="card-header">
          <h2>Dataset Overview</h2>
          <p className="subtitle">Summary of your uploaded dataset</p>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-value">{analysis.rows?.toLocaleString() || 0}</div>
              <div className="metric-label">Total Rows</div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-value">{analysis.columns?.length || 0}</div>
              <div className="metric-label">Total Columns</div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-value">{analysis.numeric_columns?.length || 0}</div>
              <div className="metric-label">Numeric Columns</div>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-content">
              <div className="metric-value">{analysis.categorical_columns?.length || 0}</div>
              <div className="metric-label">Categorical Columns</div>
            </div>
          </div>
        </div>

        <div className="columns-section">
          <div className="column-list">
            <h4>Numeric Columns</h4>
            <div className="tags">
              {analysis.numeric_columns?.map((col) => (
                <span key={col} className="tag numeric">
                  {col}
                </span>
              ))}
            </div>
          </div>

          <div className="column-list">
            <h4>Categorical Columns</h4>
            <div className="tags">
              {analysis.categorical_columns?.map((col) => (
                <span key={col} className="tag categorical">
                  {col}
                </span>
              ))}
            </div>
          </div>
        </div>

        {analysis.basic_stats && (
          <div className="stats-section">
            <h3>Statistical Summary</h3>
            <div className="stats-grid">
              {Object.entries(analysis.basic_stats).map(([column, stats]) => (
                <div key={column} className="stat-card">
                  <h4>{column}</h4>
                  <div className="stat-values">
                    <div className="stat-item">
                      <span>Mean</span>
                      <strong>{stats.mean?.toFixed(2)}</strong>
                    </div>
                    <div className="stat-item">
                      <span>Median</span>
                      <strong>{stats.median?.toFixed(2)}</strong>
                    </div>
                    <div className="stat-item">
                      <span>Std Dev</span>
                      <strong>{stats.std?.toFixed(2)}</strong>
                    </div>
                    <div className="stat-item">
                      <span>Min</span>
                      <strong>{stats.min?.toFixed(2)}</strong>
                    </div>
                    <div className="stat-item">
                      <span>Max</span>
                      <strong>{stats.max?.toFixed(2)}</strong>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataSummary;
