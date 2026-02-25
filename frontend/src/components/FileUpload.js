import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import api from '../services/api';

const FileUpload = ({ onUploadSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const onDrop = useCallback(
    async (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setLoading(true);
      setMessage('Uploading file...');

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await api.post('/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        onUploadSuccess(response.data.session_id, response.data.summary);
        setMessage('File uploaded successfully.');
      } catch (error) {
        setMessage(`Error: ${error.response?.data?.error || 'Upload failed'}`);
      } finally {
        setLoading(false);
      }
    },
    [onUploadSuccess]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
  });

  return (
    <div className="container">
      <div className="card">
        <div className="card-header">
          <h2>Upload Data File</h2>
          <p className="subtitle">Supported formats: CSV, JSON (Max 100MB)</p>
        </div>

        <div {...getRootProps()} className={`upload-zone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <div className="upload-content">
            {loading ? (
              <div className="loading-spinner large"></div>
            ) : isDragActive ? (
              <>
                <div className="upload-icon">DROP</div>
                <p>Drop the file here...</p>
              </>
            ) : (
              <>
                <div className="upload-icon">FILE</div>
                <p>Drag and drop a file here, or click to select</p>
                <small>Supports CSV and JSON files</small>
              </>
            )}
          </div>
        </div>

        {message && (
          <div className={`alert ${message.startsWith('Error:') ? 'error' : 'success'}`}>
            {message}
          </div>
        )}

        <div className="sample-section">
          <h3>Quick Start</h3>
          <p>Copy this sample data into a CSV file to test:</p>
          <div className="code-block">
            {`age,income,spending_score,country,category
25,15000,39,USA,A
35,25000,81,UK,B
45,35000,6,Canada,A
55,45000,77,USA,C
30,55000,40,UK,B
40,75000,76,Canada,A
28,35000,25,USA,B
50,80000,90,UK,C
33,45000,45,Canada,A`}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
