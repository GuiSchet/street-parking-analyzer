import { useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useParkingStore } from './stores/parkingStore';
import ParkingMap from './components/Map/ParkingMap';
import Stats from './components/Dashboard/Stats';
import SpaceList from './components/Dashboard/SpaceList';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/parking';

function App() {
  const { data: wsData, connected } = useWebSocket(WS_URL);
  const { spaces, stats, setSpaces, applyChanges } = useParkingStore();

  // Cargar estado inicial
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/spaces`);
        setSpaces(response.data.spaces);
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };

    fetchInitialData();
  }, [setSpaces]);

  // Procesar mensajes WebSocket
  useEffect(() => {
    if (!wsData) return;

    if (wsData.type === 'initial_state') {
      setSpaces(wsData.spaces);
    } else if (wsData.type === 'parking_update') {
      applyChanges(wsData.changes);
    }
  }, [wsData, setSpaces, applyChanges]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Street Parking Analyzer</h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-gray-400">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </header>

        <Stats stats={stats} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h2 className="text-2xl font-semibold mb-4">Live Parking Map</h2>
              <ParkingMap spaces={spaces} width={800} height={600} />
            </div>
          </div>

          <div className="lg:col-span-1">
            <SpaceList spaces={spaces} />
          </div>
        </div>

        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Street Parking Analyzer v1.0.0</p>
          <p className="mt-1">Real-time parking space detection using YOLO</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
