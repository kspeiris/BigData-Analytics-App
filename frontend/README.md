# Big Data Analysis Web App

A React + Flask web application for big data analysis without database dependency.

## Features

- 📊 Upload and analyze CSV/JSON files
- 🔍 Basic statistics and data exploration
- 📈 Correlation analysis and matrix visualization
- 🎯 K-means clustering
- 📉 Trend analysis and visualization
- 🎨 Interactive charts and plots
- 💾 No database required - all data processed in memory

## Tech Stack

- **Frontend**: React, Recharts, Axios
- **Backend**: Flask, Pandas, Scikit-learn, Dask
- **Processing**: In-memory data processing

## Setup Instructions

### Backend Setup
```bash
cd backend
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py