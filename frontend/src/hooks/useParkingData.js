import { useEffect, useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useParkingData = () => {
  const [spaces, setSpaces] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    available: 0,
    occupied: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchSpaces = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/spaces`);
      setSpaces(response.data.spaces);
      setStats({
        total: response.data.total,
        available: response.data.available,
        occupied: response.data.occupied
      });
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSpaces();
  }, []);

  return { spaces, stats, loading, error, refetch: fetchSpaces };
};
