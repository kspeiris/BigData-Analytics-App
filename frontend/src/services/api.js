import axios from 'axios';

const API_BASE = 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    if (error.response?.data?.error) {
      alert(`API Error: ${error.response.data.error}`);
    }
    return Promise.reject(error);
  }
);

export default api;
