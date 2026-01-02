import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const parkingAPI = {
  // Espacios
  getSpaces: () => api.get('/api/spaces'),
  getSpace: (spaceId) => api.get(`/api/spaces/${spaceId}`),
  getSpaceHistory: (spaceId, hours = 24) => api.get(`/api/spaces/${spaceId}/history`, { params: { hours } }),

  // Configuración
  getConfig: () => api.get('/api/config'),
  saveCalibration: (config) => api.post('/api/config/calibration', config),
  updateParams: (params) => api.put('/api/config/params', params),

  // Analíticas
  getAnalytics: (days = 7) => api.get('/api/analytics', { params: { days } }),
  getOccupancy: () => api.get('/api/analytics/occupancy'),

  // Vehículos
  getActiveVehicles: () => api.get('/api/vehicles/active'),

  // Sistema
  getHealth: () => api.get('/api/health'),
  getLogs: (level = null, limit = 100) => api.get('/api/logs', { params: { level, limit } })
};

export default api;
