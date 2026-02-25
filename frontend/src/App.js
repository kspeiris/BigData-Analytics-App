import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import DataSummary from './components/DataSummary';
import AnalysisTools from './components/AnalysisTools';
import AnalysisCharts from './components/AnalysisCharts';
import './App.css';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [analysisKey, setAnalysisKey] = useState(0);

  const handleUploadSuccess = (nextSessionId, nextAnalysis) => {
    setSessionId(nextSessionId);
    setAnalysis(nextAnalysis);
    setActiveTab('summary');
    setAnalysisKey((v) => v + 1);
  };

  const clearSession = () => {
    setSessionId(null);
    setAnalysis(null);
    setActiveTab('upload');
    setAnalysisKey((v) => v + 1);
  };

  return (
    <div className="app-shell">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48" role="img">
                <rect x="8" y="24" width="6" height="14" rx="2"></rect>
                <rect x="18" y="18" width="6" height="20" rx="2"></rect>
                <rect x="28" y="12" width="6" height="26" rx="2"></rect>
                <path d="M10 14 C16 9, 23 9, 30 6 C33 5, 36 6, 38 8"></path>
                <circle cx="38" cy="8" r="3"></circle>
              </svg>
            </div>
            <div className="logo-text">
              <h1>DataAnalyzer Pro</h1>
              <span>Big Data Analysis Platform</span>
            </div>
          </div>
          {sessionId && (
            <button onClick={clearSession} className="btn btn-outline">
              New Session
            </button>
          )}
        </div>
      </header>

      <nav className="navigation">
        <div className="nav-content">
          <button
            onClick={() => setActiveTab('upload')}
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
          >
            <span className="nav-icon">U</span>
            <span className="nav-text">Upload</span>
          </button>

          {sessionId && (
            <>
              <button
                onClick={() => setActiveTab('summary')}
                className={`nav-item ${activeTab === 'summary' ? 'active' : ''}`}
              >
                <span className="nav-icon">O</span>
                <span className="nav-text">Overview</span>
              </button>
              <button
                onClick={() => setActiveTab('analysis')}
                className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`}
              >
                <span className="nav-icon">A</span>
                <span className="nav-text">Analysis</span>
              </button>
              <button
                onClick={() => setActiveTab('charts')}
                className={`nav-item ${activeTab === 'charts' ? 'active' : ''}`}
              >
                <span className="nav-icon">C</span>
                <span className="nav-text">Charts</span>
              </button>
            </>
          )}
        </div>
      </nav>

      <main className="main-content">
        {activeTab === 'upload' && <FileUpload onUploadSuccess={handleUploadSuccess} />}
        {activeTab === 'summary' && sessionId && <DataSummary analysis={analysis} />}
        {activeTab === 'analysis' && sessionId && (
          <AnalysisTools key={analysisKey} sessionId={sessionId} analysis={analysis} />
        )}
        {activeTab === 'charts' && sessionId && <AnalysisCharts sessionId={sessionId} />}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p>(c) 2024 DataAnalyzer Pro | Big Data Analysis Platform</p>
          <div className="footer-links">
            <span>React + Flask</span>
            <span>|</span>
            <span>No Database Required</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
