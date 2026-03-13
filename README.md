# DataAnalyzer Pro

Professional big data analysis platform for local CSV/JSON exploration with React and Flask.

![hero image](Bigdatahero.png)

## Key Features

- Smart upload for `.csv` and `.json`
- Disk-backed dataset sessions instead of fully in-memory storage
- CSV uploads supported up to 2 GB
- Basic dataset summary: rows, columns, dtypes, missing values
- Core analytics:
  - Correlation matrix
  - K-means clustering
  - Scatter visualization
  - Trend analysis
  - Outlier detection
  - PCA
- Advanced charts dashboard with linked filters
- Sampling-based heavy analysis paths for large datasets

## Tech Stack

- Frontend: React, Axios, Recharts, React Dropzone
- Backend: Flask, Pandas, NumPy, Scikit-learn, Matplotlib
- Data/Compute: disk-backed uploads, Pandas, Dask

## Project Structure

```text
Bigdata-analysis-app/
|-- backend/
|   |-- app.py
|   `-- requirements_new.txt
|-- frontend/
|   |-- public/
|   `-- src/
|       |-- components/
|       |-- services/
|       `-- App.js
`-- sample_data/
```

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements_new.txt
python app.py
```

Backend URL: `http://localhost:5000`

### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend URL: `http://localhost:3000`

## Main API Endpoints

- `GET /health`
- `POST /upload`
- `POST /analyze/basic`
- `POST /analyze/correlation`
- `POST /analyze/clustering`
- `POST /visualize/scatter`
- `POST /analyze/trend`
- `POST /analyze/outliers`
- `POST /analyze/pca`
- `POST /analyze/charts`
- `POST /debug/dataset`
- `POST /session/clear`

## Usage Flow

1. Start the backend.
2. Start the frontend.
3. Upload a CSV or JSON file.
4. Review the summary in the dashboard.
5. Run analysis tools.
6. Explore the charts tab.

## Notes

- Large CSV uploads are stored on disk under backend session folders.
- Heavy endpoints use bounded sampling for scalability on large datasets.
- This improves handling for large local datasets, but it is not a full distributed big data platform.
- Frontend API timeout is set to 10 minutes for long-running uploads.

## Troubleshooting

- If the frontend shows connection errors, start the backend first.
- If you get `Session not found`, re-upload the dataset.
- If dependencies are missing, reinstall from `backend/requirements_new.txt` and run `npm install` in `frontend/`.
