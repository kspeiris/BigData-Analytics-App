import React, { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import api from '../services/api';

const PIE_COLORS = ['#0a6cff', '#0ea772', '#d97706', '#0284c7', '#dc2626', '#7c3aed', '#14b8a6', '#64748b'];

const AnalysisCharts = ({ sessionId }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [activePairIndex, setActivePairIndex] = useState(0);
  const [activeDistributionIndex, setActiveDistributionIndex] = useState(0);
  const [activeCategoricalIndex, setActiveCategoricalIndex] = useState(0);
  const [activeBoxIndex, setActiveBoxIndex] = useState(0);
  const [rangeColumn, setRangeColumn] = useState('');
  const [rangeMin, setRangeMin] = useState('');
  const [rangeMax, setRangeMax] = useState('');
  const [categoryColumn, setCategoryColumn] = useState('');
  const [categoryValue, setCategoryValue] = useState('');
  const [dateColumn, setDateColumn] = useState('');
  const [dateStart, setDateStart] = useState('');
  const [dateEnd, setDateEnd] = useState('');

  const buildFiltersPayload = () => ({
    range: { column: rangeColumn || null, min: rangeMin, max: rangeMax },
    category: { column: categoryColumn || null, value: categoryValue || '' },
    date: { column: dateColumn || null, start: dateStart || '', end: dateEnd || '' },
  });

  const loadChartsWithPayload = async (filtersPayload) => {
    if (!sessionId) return;
    setLoading(true);
    setError('');
    try {
      const response = await api.post('/analyze/charts', {
        session_id: sessionId,
        filters: filtersPayload,
      });
      setData(response.data);
      setActivePairIndex(0);
      setActiveDistributionIndex(0);
      setActiveCategoricalIndex(0);
      setActiveBoxIndex(0);
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Failed to load charts');
    } finally {
      setLoading(false);
    }
  };

  const loadCharts = async () => {
    await loadChartsWithPayload(buildFiltersPayload());
  };

  const clearFilters = async () => {
    setRangeColumn('');
    setRangeMin('');
    setRangeMax('');
    setCategoryColumn('');
    setCategoryValue('');
    setDateColumn('');
    setDateStart('');
    setDateEnd('');
    await loadChartsWithPayload({
      range: { column: null, min: '', max: '' },
      category: { column: null, value: '' },
      date: { column: null, start: '', end: '' },
    });
  };

  useEffect(() => {
    loadCharts();
  }, [sessionId]);

  useEffect(() => {
    const meta = data?.filter_metadata;
    if (!meta) return;

    if (!rangeColumn && Array.isArray(meta.numeric_columns) && meta.numeric_columns.length > 0) {
      const first = meta.numeric_columns[0];
      setRangeColumn(first.column);
      setRangeMin(first.min);
      setRangeMax(first.max);
    } else if (rangeColumn && Array.isArray(meta.numeric_columns)) {
      const current = meta.numeric_columns.find((n) => n.column === rangeColumn);
      if (!current) {
        setRangeColumn('');
        setRangeMin('');
        setRangeMax('');
      }
    }

    if (!categoryColumn && Array.isArray(meta.categorical_columns) && meta.categorical_columns.length > 0) {
      setCategoryColumn(meta.categorical_columns[0].column);
      setCategoryValue('');
    } else if (categoryColumn && Array.isArray(meta.categorical_columns)) {
      const current = meta.categorical_columns.find((c) => c.column === categoryColumn);
      if (!current) {
        setCategoryColumn('');
        setCategoryValue('');
      }
    }

    if (!dateColumn && Array.isArray(meta.date_columns) && meta.date_columns.length > 0) {
      setDateColumn(meta.date_columns[0].column);
    } else if (dateColumn && Array.isArray(meta.date_columns)) {
      const current = meta.date_columns.find((d) => d.column === dateColumn);
      if (!current) {
        setDateColumn('');
        setDateStart('');
        setDateEnd('');
      }
    }
  }, [data]);

  const activeDistribution = useMemo(
    () => data?.distributions?.[activeDistributionIndex] || null,
    [data, activeDistributionIndex]
  );
  const activePair = useMemo(() => data?.scatter_pairs?.[activePairIndex] || null, [data, activePairIndex]);
  const activeCategory = useMemo(
    () => data?.categorical_breakdown?.[activeCategoricalIndex] || null,
    [data, activeCategoricalIndex]
  );
  const activeBoxStats = useMemo(() => data?.boxplot_stats?.[activeBoxIndex] || null, [data, activeBoxIndex]);

  const correlationLeaderboard = useMemo(() => {
    if (!data?.correlation?.columns?.length || !data?.correlation?.matrix?.length) return [];
    const cols = data.correlation.columns;
    const matrix = data.correlation.matrix;
    const pairs = [];
    for (let i = 0; i < cols.length; i += 1) {
      for (let j = i + 1; j < cols.length; j += 1) {
        const value = Number(matrix?.[i]?.[j]);
        if (!Number.isNaN(value)) {
          pairs.push({
            pair: `${cols[i]} vs ${cols[j]}`,
            correlation: value,
            absCorrelation: Math.abs(value),
          });
        }
      }
    }
    return pairs.sort((a, b) => b.absCorrelation - a.absCorrelation).slice(0, 10);
  }, [data]);

  const pcaCumulative = useMemo(() => {
    if (!Array.isArray(data?.pca_variance)) return [];
    let running = 0;
    return data.pca_variance.map((item) => {
      running += Number(item.variance_percent || 0);
      return { component: item.component, cumulative: Number(running.toFixed(2)) };
    });
  }, [data]);

  const completenessData = useMemo(() => {
    const rows = Number(data?.summary?.rows || 0);
    const cols = Number(data?.summary?.columns || 0);
    const totalCells = rows * cols;
    const missing = Number(data?.summary?.total_missing_cells || 0);
    const present = Math.max(0, totalCells - missing);
    return [
      { name: 'Present', value: present },
      { name: 'Missing', value: missing },
    ];
  }, [data]);

  const selectedRangeMeta = useMemo(
    () => data?.filter_metadata?.numeric_columns?.find((n) => n.column === rangeColumn),
    [data, rangeColumn]
  );
  const selectedCategoryMeta = useMemo(
    () => data?.filter_metadata?.categorical_columns?.find((c) => c.column === categoryColumn),
    [data, categoryColumn]
  );

  return (
    <div className="container">
      <div className="card charts-card">
        <div className="card-header charts-header">
          <div>
            <h2>Advanced Charts & Insights</h2>
            <p className="subtitle">High-detail analytics dashboard with multi-view visual diagnostics.</p>
          </div>
          <button className="btn btn-primary charts-refresh" onClick={loadCharts} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh Charts'}
          </button>
        </div>

        {data?.filter_metadata && (
          <div className="dashboard-filters">
            <div className="filter-row">
              <div className="filter-group">
                <label>Range Column</label>
                <select
                  className="select-input small"
                  value={rangeColumn}
                  onChange={(e) => {
                    const col = e.target.value;
                    setRangeColumn(col);
                    const meta = data.filter_metadata.numeric_columns?.find((n) => n.column === col);
                    if (meta) {
                      setRangeMin(meta.min);
                      setRangeMax(meta.max);
                    } else {
                      setRangeMin('');
                      setRangeMax('');
                    }
                  }}
                >
                  <option value="">None</option>
                  {(data.filter_metadata.numeric_columns || []).map((item) => (
                    <option key={item.column} value={item.column}>
                      {item.column}
                    </option>
                  ))}
                </select>
              </div>
              <div className="filter-group">
                <label>Min</label>
                <input
                  className="select-input small"
                  type="number"
                  step="any"
                  value={rangeMin}
                  onChange={(e) => setRangeMin(e.target.value)}
                  placeholder={selectedRangeMeta ? String(selectedRangeMeta.min) : ''}
                />
              </div>
              <div className="filter-group">
                <label>Max</label>
                <input
                  className="select-input small"
                  type="number"
                  step="any"
                  value={rangeMax}
                  onChange={(e) => setRangeMax(e.target.value)}
                  placeholder={selectedRangeMeta ? String(selectedRangeMeta.max) : ''}
                />
              </div>
            </div>

            <div className="filter-row">
              <div className="filter-group">
                <label>Category Column</label>
                <select
                  className="select-input small"
                  value={categoryColumn}
                  onChange={(e) => {
                    setCategoryColumn(e.target.value);
                    setCategoryValue('');
                  }}
                >
                  <option value="">None</option>
                  {(data.filter_metadata.categorical_columns || []).map((item) => (
                    <option key={item.column} value={item.column}>
                      {item.column}
                    </option>
                  ))}
                </select>
              </div>
              <div className="filter-group">
                <label>Category Value</label>
                <select
                  className="select-input small"
                  value={categoryValue}
                  onChange={(e) => setCategoryValue(e.target.value)}
                >
                  <option value="">All</option>
                  {(selectedCategoryMeta?.values || []).map((val) => (
                    <option key={val} value={val}>
                      {val}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="filter-row">
              <div className="filter-group">
                <label>Date Column</label>
                <select
                  className="select-input small"
                  value={dateColumn}
                  onChange={(e) => setDateColumn(e.target.value)}
                >
                  <option value="">None</option>
                  {(data.filter_metadata.date_columns || []).map((item) => (
                    <option key={item.column} value={item.column}>
                      {item.column}
                    </option>
                  ))}
                </select>
              </div>
              <div className="filter-group">
                <label>Start</label>
                <input
                  className="select-input small"
                  type="date"
                  value={dateStart}
                  onChange={(e) => setDateStart(e.target.value)}
                />
              </div>
              <div className="filter-group">
                <label>End</label>
                <input
                  className="select-input small"
                  type="date"
                  value={dateEnd}
                  onChange={(e) => setDateEnd(e.target.value)}
                />
              </div>
            </div>

            <div className="filter-actions">
              <button className="btn btn-primary" onClick={loadCharts} disabled={loading}>
                {loading ? 'Applying...' : 'Apply Linked Filters'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={clearFilters}
              >
                Clear Filters
              </button>
            </div>
          </div>
        )}

        {error && <div className="alert error">{error}</div>}
        {loading && !data && (
          <div className="charts-loading">
            <div className="loading-spinner"></div>
            <p>Computing analytics views...</p>
          </div>
        )}

        {data && (
          <div className="charts-layout">
            <div className="kpi-grid">
              <div className="kpi-card"><span>Rows</span><strong>{data.summary?.rows ?? 0}</strong></div>
              <div className="kpi-card"><span>Columns</span><strong>{data.summary?.columns ?? 0}</strong></div>
              <div className="kpi-card"><span>Numeric</span><strong>{data.summary?.numeric_columns ?? 0}</strong></div>
              <div className="kpi-card"><span>Categorical</span><strong>{data.summary?.categorical_columns ?? 0}</strong></div>
              <div className="kpi-card"><span>Missing Cells</span><strong>{data.summary?.total_missing_cells ?? 0}</strong></div>
              <div className="kpi-card"><span>Rows Before Filter</span><strong>{data.summary?.rows_before_filter ?? data.summary?.rows ?? 0}</strong></div>
              <div className="kpi-card"><span>Rows After Filter</span><strong>{data.summary?.rows_after_filter ?? data.summary?.rows ?? 0}</strong></div>
            </div>

            <div className="charts-grid">
              <section className="chart-panel span-2">
                <h3>Missing Value Heat</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={(data.missing_values || []).slice(0, 12)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="column" interval={0} angle={-30} textAnchor="end" height={70} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="missing" fill="#dc2626" name="Missing Count" />
                    <Bar dataKey="missing_percent" fill="#0284c7" name="Missing %" />
                  </BarChart>
                </ResponsiveContainer>
              </section>

              <section className="chart-panel">
                <h3>PCA Explained Variance</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={data.pca_variance || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="component" />
                    <YAxis />
                    <Tooltip formatter={(v) => [`${Number(v).toFixed(2)}%`, 'Variance %']} />
                    <Bar dataKey="variance_percent" fill="#0a6cff" />
                  </BarChart>
                </ResponsiveContainer>
              </section>

              <section className="chart-panel">
                <h3>PCA Cumulative Variance</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={pcaCumulative}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="component" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(v) => [`${Number(v).toFixed(2)}%`, 'Cumulative']} />
                    <Line type="monotone" dataKey="cumulative" stroke="#0ea772" strokeWidth={2.5} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </section>

              <section className="chart-panel span-2">
                <h3>Numeric Distribution Explorer</h3>
                <div className="chart-toolbar">
                  <label>Column</label>
                  <select
                    className="select-input small"
                    value={activeDistributionIndex}
                    onChange={(e) => setActiveDistributionIndex(parseInt(e.target.value, 10))}
                  >
                    {(data.distributions || []).map((d, idx) => (
                      <option key={d.column} value={idx}>
                        {d.column}
                      </option>
                    ))}
                  </select>
                </div>
                {activeDistribution ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={activeDistribution.bins || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" hide />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="count" stroke="#0a6cff" fill="#93c5fd" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="empty-chart-text">No distribution data available.</p>
                )}
              </section>

              <section className="chart-panel span-2">
                <h3>Pairwise Scatter Explorer</h3>
                <div className="chart-toolbar">
                  <label>Pair</label>
                  <select
                    className="select-input small"
                    value={activePairIndex}
                    onChange={(e) => setActivePairIndex(parseInt(e.target.value, 10))}
                  >
                    {(data.scatter_pairs || []).map((pair, idx) => (
                      <option key={`${pair.x_column}-${pair.y_column}`} value={idx}>
                        {pair.x_column} vs {pair.y_column}
                      </option>
                    ))}
                  </select>
                </div>
                {activePair ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="x" name={activePair.x_column} />
                      <YAxis type="number" dataKey="y" name={activePair.y_column} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter data={activePair.points || []} fill="#0ea772" />
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="empty-chart-text">No scatter pairs available.</p>
                )}
              </section>

              <section className="chart-panel span-2">
                <h3>Correlation Leaderboard</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={correlationLeaderboard} layout="vertical" margin={{ left: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} />
                    <YAxis type="category" width={180} dataKey="pair" />
                    <Tooltip formatter={(v) => [Number(v).toFixed(3), '|Correlation|']} />
                    <Bar dataKey="absCorrelation" fill="#7c3aed" />
                  </BarChart>
                </ResponsiveContainer>
              </section>

              <section className="chart-panel">
                <h3>Categorical Breakdown</h3>
                <div className="chart-toolbar">
                  <label>Column</label>
                  <select
                    className="select-input small"
                    value={activeCategoricalIndex}
                    onChange={(e) => setActiveCategoricalIndex(parseInt(e.target.value, 10))}
                  >
                    {(data.categorical_breakdown || []).map((item, idx) => (
                      <option key={item.column} value={idx}>
                        {item.column}
                      </option>
                    ))}
                  </select>
                </div>
                {activeCategory ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                      <Pie data={activeCategory.values || []} dataKey="count" nameKey="category" outerRadius={90} label>
                        {(activeCategory.values || []).map((_, idx) => (
                          <Cell key={`cell-${idx}`} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="empty-chart-text">No categorical chart data available.</p>
                )}
              </section>

              <section className="chart-panel">
                <h3>Dataset Completeness</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie data={completenessData} dataKey="value" nameKey="name" outerRadius={90} label>
                      {completenessData.map((_, idx) => (
                        <Cell key={`completeness-${idx}`} fill={idx === 0 ? '#0ea772' : '#dc2626'} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v) => [Number(v).toLocaleString(), 'Cells']} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </section>

              <section className="chart-panel span-2">
                <h3>Box-Stat Explorer</h3>
                <div className="chart-toolbar">
                  <label>Column</label>
                  <select
                    className="select-input small"
                    value={activeBoxIndex}
                    onChange={(e) => setActiveBoxIndex(parseInt(e.target.value, 10))}
                  >
                    {(data.boxplot_stats || []).map((item, idx) => (
                      <option key={item.column} value={idx}>
                        {item.column}
                      </option>
                    ))}
                  </select>
                </div>
                {activeBoxStats ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart
                      data={[
                        { stat: 'Min', value: activeBoxStats.min },
                        { stat: 'Q1', value: activeBoxStats.q1 },
                        { stat: 'Median', value: activeBoxStats.median },
                        { stat: 'Q3', value: activeBoxStats.q3 },
                        { stat: 'Max', value: activeBoxStats.max },
                        { stat: 'Mean', value: activeBoxStats.mean },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="stat" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#0a6cff" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="empty-chart-text">No box-stat data available.</p>
                )}
              </section>

              <section className="chart-panel span-2">
                <h3>Trend Lines (Top Numeric Columns)</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="index" type="number" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {(data.trend_series || []).map((series, idx) => (
                      <Line
                        key={series.column}
                        data={series.points || []}
                        dataKey="value"
                        name={series.column}
                        stroke={PIE_COLORS[idx % PIE_COLORS.length]}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </section>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisCharts;
