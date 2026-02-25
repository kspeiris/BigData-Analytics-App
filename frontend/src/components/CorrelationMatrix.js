import React from 'react';

const CorrelationMatrix = ({ data }) => {
  if (!data) return null;

  const getColorIntensity = (value) => {
    const intensity = Math.abs(value);
    const hue = value > 0 ? 210 : 0;
    return `hsl(${hue}, 70%, ${90 - intensity * 40}%)`;
  };

  return (
    <div className="correlation-matrix">
      <h2>Correlation Matrix</h2>
      
      <div className="matrix-container">
        <table className="correlation-table">
          <thead>
            <tr>
              <th></th>
              {data.columns.map(column => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.matrix.map((row, i) => (
              <tr key={i}>
                <td className="row-header"><strong>{data.columns[i]}</strong></td>
                {row.map((cell, j) => (
                  <td 
                    key={j}
                    className="correlation-cell"
                    style={{
                      backgroundColor: getColorIntensity(cell),
                      color: Math.abs(cell) > 0.5 ? 'white' : 'black'
                    }}
                    title={`${data.columns[i]} vs ${data.columns[j]}: ${cell}`}
                  >
                    {cell.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.strong_correlations && data.strong_correlations.length > 0 && (
        <div className="strong-correlations">
          <h3>Strong Correlations (|r| &gt 0.7)</h3>
          <div className="correlation-list">
            {data.strong_correlations.map((corr, index) => (
              <div key={index} className="correlation-item">
                <span className="correlation-pair">
                  {corr.column1} â†” {corr.column2}
                </span>
                <span className={`correlation-value ${corr.correlation > 0 ? 'positive' : 'negative'}`}>
                  {corr.correlation > 0 ? 'â†‘' : 'â†“'} {corr.correlation.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="correlation-legend">
        <div className="legend-item">
          <div className="legend-color" style={{backgroundColor: 'hsl(0, 70%, 50%)'}}></div>
          <span>Strong Negative</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{backgroundColor: 'hsl(0, 70%, 90%)'}}></div>
          <span>Weak Negative</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{backgroundColor: 'hsl(210, 70%, 90%)'}}></div>
          <span>Weak Positive</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{backgroundColor: 'hsl(210, 70%, 50%)'}}></div>
          <span>Strong Positive</span>
        </div>
      </div>
    </div>
  );
};

export default CorrelationMatrix;
