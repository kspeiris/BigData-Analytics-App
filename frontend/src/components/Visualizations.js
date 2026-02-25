import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, ResponsiveContainer } from 'recharts';

const Visualizations = ({ charts }) => {
  if (charts.length === 0) {
    return (
      <div className="visualizations">
        <h2>Visualizations</h2>
        <p>No charts generated yet. Run some analyses to see visualizations here.</p>
      </div>
    );
  }

  return (
    <div className="visualizations">
      <h2>Data Visualizations</h2>
      
      <div className="charts-grid">
        {charts.map((chart, index) => (
          <div key={index} className="chart-container">
            <h3>{chart.title}</h3>
            
            {chart.type === 'scatter' && (
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={chart.data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="x" 
                    type="number"
                    name={chart.columns?.[0]}
                    label={{ value: chart.columns?.[0], position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    dataKey="y" 
                    type="number"
                    name={chart.columns?.[1]}
                    label={{ value: chart.columns?.[1], angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter dataKey="y" fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            )}

            {chart.type === 'trend' && (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chart.data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#8884d8" 
                    name="Actual" 
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="trend" 
                    stroke="#82ca9d" 
                    name="Trend" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}

            {chart.type === 'image' && (
              <div className="image-chart">
                <img src={chart.image} alt={chart.title} />
              </div>
            )}

            {chart.direction && (
              <div className="chart-info">
                <p>Trend: <strong>{chart.direction}</strong></p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Visualizations;