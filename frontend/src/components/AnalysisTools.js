import React, { useEffect, useState } from 'react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import api from '../services/api';

const AnalysisTools = ({ sessionId, analysis }) => {
  const [toolsLoading, setToolsLoading] = useState(false);
  const [toolsMessage, setToolsMessage] = useState('');
  const [selectedX, setSelectedX] = useState('');
  const [selectedY, setSelectedY] = useState('');
  const [nClusters, setNClusters] = useState(3);
  const [trendColumn, setTrendColumn] = useState('');

  const [basicResult, setBasicResult] = useState(null);
  const [correlationResult, setCorrelationResult] = useState(null);
  const [clusteringResult, setClusteringResult] = useState(null);
  const [scatterResult, setScatterResult] = useState(null);
  const [debugResult, setDebugResult] = useState(null);
  const [trendResult, setTrendResult] = useState(null);
  const [outlierResult, setOutlierResult] = useState(null);
  const [pcaResult, setPcaResult] = useState(null);

  useEffect(() => {
    if (analysis?.numeric_columns?.length >= 2) {
      setSelectedX(analysis.numeric_columns[0]);
      setSelectedY(analysis.numeric_columns[1]);
    }
    if (analysis?.numeric_columns?.length >= 1) {
      setTrendColumn(analysis.numeric_columns[0]);
    }
  }, [analysis]);

  const runAnalysis = async (endpoint, analysisName, data = {}) => {
    setToolsLoading(true);
    setToolsMessage(`Running ${analysisName}...`);
    try {
      const response = await api.post(endpoint, { session_id: sessionId, ...data });
      const result = response.data;
      setToolsMessage(`${analysisName} completed successfully.`);

      if (analysisName === 'Basic Analysis') setBasicResult(result);
      if (analysisName === 'Correlation Analysis') setCorrelationResult(result);
      if (analysisName === 'Clustering Analysis') setClusteringResult(result);
      if (analysisName === 'Scatter Plot') setScatterResult(result);
      if (analysisName === 'Debug Dataset') setDebugResult(result);
      if (analysisName === 'Trend Analysis') setTrendResult(result);
      if (analysisName === 'Outlier Detection') setOutlierResult(result);
      if (analysisName === 'PCA Analysis') setPcaResult(result);
    } catch (error) {
      setToolsMessage(`Error in ${analysisName}: ${error.response?.data?.error || error.message}`);
    } finally {
      setToolsLoading(false);
    }
  };

  const getClusterDistribution = (clusters) => {
    if (!Array.isArray(clusters) || clusters.length === 0) return [];
    const counts = clusters.reduce((acc, c) => {
      const key = String(c);
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts)
      .map(([cluster, count]) => ({ cluster, count }))
      .sort((a, b) => b.count - a.count);
  };

  return (
    <div className="container">
      <div className="card">
        <div className="card-header">
          <h2>Analysis Tools</h2>
          <p className="subtitle">Run analytics and review results inline.</p>
        </div>

        {toolsMessage && (
          <div className={`alert ${toolsMessage.startsWith('Error') ? 'error' : 'success'}`}>{toolsMessage}</div>
        )}

        {analysis?.numeric_columns?.length >= 2 && (
          <div className="selection-panel">
            <h4>Column Selection</h4>
            <div className="selection-controls">
              <div className="select-group">
                <label>X Axis</label>
                <select value={selectedX} onChange={(e) => setSelectedX(e.target.value)} className="select-input">
                  {analysis.numeric_columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              <div className="select-group">
                <label>Y Axis</label>
                <select value={selectedY} onChange={(e) => setSelectedY(e.target.value)} className="select-input">
                  {analysis.numeric_columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        <div className="analysis-section">
          <h3 className="section-heading">Core Analytics</h3>
          <p className="section-subtitle">Fast, high-value checks for quick insight.</p>
          <div className="tools-grid">
            <div className="tool-card">
              <div className="tool-icon">BA</div>
              <h3>Basic Analysis</h3>
              <p>Shape, types, nulls and memory usage.</p>
              <button onClick={() => runAnalysis('/analyze/basic', 'Basic Analysis')} disabled={toolsLoading} className="btn btn-primary">
                {toolsLoading ? 'Running...' : 'Run'}
              </button>
              {basicResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Rows:</strong> {basicResult.shape?.rows ?? '-'}</div>
                    <div><strong>Columns:</strong> {basicResult.shape?.columns ?? '-'}</div>
                    <div><strong>Memory:</strong> {((basicResult.memory_usage || 0) / 1024).toFixed(2)} KB</div>
                    <div><strong>Data Types:</strong> {Object.keys(basicResult.data_types || {}).length}</div>
                    <div><strong>Missing Cells:</strong> {Object.values(basicResult.missing_values || {}).reduce((a, v) => a + Number(v || 0), 0)}</div>
                    <div><strong>Missing %:</strong> {(() => {
                      const rows = Number(basicResult.shape?.rows || 0);
                      const cols = Number(basicResult.shape?.columns || 0);
                      const total = rows * cols;
                      const missing = Object.values(basicResult.missing_values || {}).reduce((a, v) => a + Number(v || 0), 0);
                      return total > 0 ? `${((missing / total) * 100).toFixed(2)}%` : '0.00%';
                    })()}</div>
                  </div>
                </div>
              )}
            </div>

            <div className="tool-card">
              <div className="tool-icon">CM</div>
              <h3>Correlation Matrix</h3>
              <p>Find strongest numeric relationships.</p>
              <button onClick={() => runAnalysis('/analyze/correlation', 'Correlation Analysis')} disabled={toolsLoading} className="btn btn-success">
                {toolsLoading ? 'Running...' : 'Run'}
              </button>
              {correlationResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Columns:</strong> {correlationResult.columns?.length ?? 0}</div>
                    <div><strong>Strong Pairs:</strong> {correlationResult.strong_correlations?.length ?? 0}</div>
                    <div><strong>Matrix Size:</strong> {(correlationResult.columns?.length || 0)} x {(correlationResult.columns?.length || 0)}</div>
                    <div><strong>Status:</strong> {(correlationResult.strong_correlations?.length || 0) > 0 ? 'Strong relationships found' : 'No strong pairs'}</div>
                  </div>
                  {Array.isArray(correlationResult.strong_correlations) && correlationResult.strong_correlations.length > 0 && (
                    <div className="inline-list">
                      {correlationResult.strong_correlations.slice(0, 3).map((item, idx) => (
                        <div key={`${item.column1}-${item.column2}-${idx}`} className="inline-list-item">
                          <span>{item.column1} vs {item.column2}</span>
                          <strong>{Number(item.correlation).toFixed(3)}</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="tool-card">
              <div className="tool-icon">SP</div>
              <h3>Scatter Plot</h3>
              <p>Visual check for pairwise patterns.</p>
              <button
                onClick={() =>
                  runAnalysis('/visualize/scatter', 'Scatter Plot', {
                    x_column: selectedX,
                    y_column: selectedY,
                  })
                }
                disabled={toolsLoading || !selectedX || !selectedY}
                className="btn btn-danger"
              >
                {toolsLoading ? 'Running...' : 'Run'}
              </button>
              {scatterResult?.image && (
                <div className="inline-result">
                  <div className="inline-title">Latest Plot</div>
                  <div className="inline-grid two">
                    <div><strong>X Axis:</strong> {selectedX || '-'}</div>
                    <div><strong>Y Axis:</strong> {selectedY || '-'}</div>
                  </div>
                  {scatterResult.message && <div className="inline-note">{scatterResult.message}</div>}
                  <img src={scatterResult.image} alt="Scatter Plot" className="inline-plot" />
                </div>
              )}
            </div>

            <div className="tool-card">
              <div className="tool-icon">KM</div>
              <h3>Clustering</h3>
              <p>K-means segmentation.</p>
              <div className="clustering-controls">
                <label>Clusters</label>
                <select className="select-input small" value={nClusters} onChange={(e) => setNClusters(parseInt(e.target.value, 10))}>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                </select>
              </div>
              <button
                onClick={() => runAnalysis('/analyze/clustering', 'Clustering Analysis', { n_clusters: nClusters })}
                disabled={toolsLoading}
                className="btn btn-warning"
              >
                {toolsLoading ? 'Running...' : 'Run'}
              </button>
              {clusteringResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Clusters:</strong> {clusteringResult.centers?.length ?? 0}</div>
                    <div><strong>Points:</strong> {clusteringResult.data_points?.length ?? 0}</div>
                    <div><strong>Inertia:</strong> {typeof clusteringResult.inertia === 'number' ? clusteringResult.inertia.toFixed(2) : '-'}</div>
                    <div><strong>Columns:</strong> {(clusteringResult.columns || []).join(', ') || '-'}</div>
                  </div>
                  {getClusterDistribution(clusteringResult.clusters).length > 0 && (
                    <div className="inline-list">
                      {getClusterDistribution(clusteringResult.clusters).map((item) => (
                        <div key={item.cluster} className="inline-list-item">
                          <span>Cluster {item.cluster}</span>
                          <strong>{item.count} points</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="tool-card">
              <div className="tool-icon">DI</div>
              <h3>Dataset Inspector</h3>
              <p>Quick structural debug snapshot.</p>
              <button onClick={() => runAnalysis('/debug/dataset', 'Debug Dataset')} disabled={toolsLoading} className="btn btn-secondary">
                {toolsLoading ? 'Running...' : 'Run'}
              </button>
              {debugResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Columns:</strong> {debugResult.columns?.length ?? 0}</div>
                    <div><strong>Shape:</strong> {Array.isArray(debugResult.shape) ? `${debugResult.shape[0]} x ${debugResult.shape[1]}` : '-'}</div>
                    <div><strong>Numeric Cols:</strong> {debugResult.numeric_columns?.length ?? 0}</div>
                    <div><strong>Sample Rows:</strong> {debugResult.sample_data?.length ?? 0}</div>
                  </div>
                  {Array.isArray(debugResult.columns) && debugResult.columns.length > 0 && (
                    <div className="inline-note">
                      <strong>Columns:</strong> {debugResult.columns.slice(0, 6).join(', ')}
                      {debugResult.columns.length > 6 ? ' ...' : ''}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="tool-card">
              <div className="tool-icon">TR</div>
              <h3>Trend Analysis</h3>
              <p>Direction and strength over index.</p>
              <div className="clustering-controls">
                <label>Column</label>
                <select className="select-input small" value={trendColumn} onChange={(e) => setTrendColumn(e.target.value)}>
                  {analysis?.numeric_columns?.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => runAnalysis('/analyze/trend', 'Trend Analysis', { column: trendColumn })}
                disabled={toolsLoading || !analysis?.numeric_columns?.length}
                className="btn btn-info"
              >
                {toolsLoading ? 'Running...' : 'Run'}
              </button>
              {trendResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Direction:</strong> {trendResult.trend_direction || '-'}</div>
                    <div><strong>Slope:</strong> {typeof trendResult.slope === 'number' ? trendResult.slope.toFixed(4) : '-'}</div>
                    <div><strong>Correlation:</strong> {typeof trendResult.correlation === 'number' ? trendResult.correlation.toFixed(4) : '-'}</div>
                    <div><strong>Data Points:</strong> {trendResult.data_points?.length ?? 0}</div>
                    <div><strong>Intercept:</strong> {typeof trendResult.intercept === 'number' ? trendResult.intercept.toFixed(4) : '-'}</div>
                    <div><strong>Trend Line Points:</strong> {trendResult.trend_line?.length ?? 0}</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="analysis-section">
          <h3 className="section-heading">Advanced Analytics</h3>
          <p className="section-subtitle">Deeper diagnostics for anomalies and latent structure.</p>
          <div className="tools-grid advanced-grid">
            <div className="tool-card wide">
              <div className="tool-icon">OD</div>
              <h3>Outlier Detection</h3>
              <p>Z-score and Isolation Forest for anomaly detection.</p>
              <button
                onClick={() => runAnalysis('/analyze/outliers', 'Outlier Detection', { z_threshold: 3 })}
                disabled={toolsLoading}
                className="btn btn-secondary"
              >
                {toolsLoading ? 'Running...' : 'Detect Outliers'}
              </button>
              {outlierResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Rows with Outliers:</strong> {outlierResult.rows_with_any_outlier ?? 0} / {outlierResult.total_rows ?? 0}</div>
                    <div><strong>Outlier %:</strong> {typeof outlierResult.rows_with_any_outlier_percent === 'number' ? outlierResult.rows_with_any_outlier_percent.toFixed(2) : '0.00'}%</div>
                    <div><strong>Isolation Anomalies:</strong> {outlierResult.isolation_forest?.anomalies ?? 0}</div>
                    <div><strong>Rows Scored:</strong> {outlierResult.isolation_forest?.rows_scored ?? 0}</div>
                  </div>
                  {Array.isArray(outlierResult.column_summary) && outlierResult.column_summary.length > 0 && (
                    <div className="inline-list">
                      {outlierResult.column_summary.slice(0, 4).map((item) => (
                        <div key={item.column} className="inline-list-item">
                          <span>{item.column}</span>
                          <strong>{item.outlier_count} outliers</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="tool-card wide">
              <div className="tool-icon">PCA</div>
              <h3>PCA Analysis</h3>
              <p>Principal components and explained variance.</p>
              <button onClick={() => runAnalysis('/analyze/pca', 'PCA Analysis', { n_components: 2 })} disabled={toolsLoading} className="btn btn-primary">
                {toolsLoading ? 'Running...' : 'Run PCA'}
              </button>
              {pcaResult && (
                <div className="inline-result">
                  <div className="inline-title">Latest Result</div>
                  <div className="inline-grid two">
                    <div><strong>Rows Used:</strong> {pcaResult.rows_used ?? 0}</div>
                    <div><strong>Columns Used:</strong> {pcaResult.columns_used?.length ?? 0}</div>
                    <div><strong>PC1 Variance:</strong> {typeof pcaResult.explained_variance_ratio?.[0] === 'number' ? (pcaResult.explained_variance_ratio[0] * 100).toFixed(2) : '0.00'}%</div>
                    <div><strong>Total Variance:</strong> {typeof pcaResult.cumulative_explained_variance === 'number' ? (pcaResult.cumulative_explained_variance * 100).toFixed(2) : '0.00'}%</div>
                  </div>
                  {Array.isArray(pcaResult.feature_loadings) && pcaResult.feature_loadings.length > 0 && (
                    <div className="inline-list">
                      {pcaResult.feature_loadings
                        .slice()
                        .sort((a, b) => Math.abs(Number(b.pc1_loading || 0)) - Math.abs(Number(a.pc1_loading || 0)))
                        .slice(0, 4)
                        .map((item) => (
                          <div key={item.column} className="inline-list-item">
                            <span>{item.column}</span>
                            <strong>PC1 {Number(item.pc1_loading).toFixed(3)}</strong>
                          </div>
                        ))}
                    </div>
                  )}
                  {Array.isArray(pcaResult.points) && pcaResult.points.length > 0 && (
                    <div className="inline-chart">
                      <ResponsiveContainer width="100%" height={220}>
                        <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" dataKey="pc1" name="PC1" />
                          <YAxis type="number" dataKey="pc2" name="PC2" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter data={pcaResult.points} fill="#0a6cff" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {toolsLoading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Processing your data...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisTools;
