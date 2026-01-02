# Street Parking Analyzer

Sistema de detección de espacios de estacionamiento en tiempo real utilizando YOLO para detección de vehículos y WebSocket para actualizaciones en vivo.

## 🚀 Características

- ✅ Detección de vehículos en tiempo real usando YOLOv8
- ✅ Análisis de espacios de estacionamiento disponibles
- ✅ Interfaz web interactiva con mapa de estacionamiento
- ✅ Actualizaciones en tiempo real vía WebSocket
- ✅ Estadísticas de ocupación y analíticas
- ✅ Almacenamiento histórico en MongoDB
- ✅ API REST completa
- ✅ Filtrado temporal para reducir falsos positivos

## 📋 Requisitos Previos

- Docker y Docker Compose (recomendado)
- O bien:
  - Python 3.10+
  - Node.js 18+
  - MongoDB 7.0+

## 🛠️ Instalación

### Opción 1: Con Docker (Recomendado)

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd street-parking-analyzer
```

2. Configurar variables de entorno:
```bash
# Editar docker-compose.yml y actualizar CAMERA_URL si es necesario
cp backend/.env.example backend/.env
```

3. Iniciar los servicios:
```bash
docker-compose up -d
```

4. Instalar dependencias del frontend:
```bash
cd frontend
npm install
npm run dev
```

5. Acceder a la aplicación:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- MongoDB: localhost:27017

### Opción 2: Instalación Local

#### Backend

1. Crear entorno virtual e instalar dependencias:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

3. Iniciar MongoDB:
```bash
mongod --dbpath /path/to/data
```

4. Ejecutar el backend:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

1. Instalar dependencias:
```bash
cd frontend
npm install
```

2. Configurar variables de entorno:
```bash
# Crear archivo .env
echo "VITE_API_URL=http://localhost:8000" > .env
echo "VITE_WS_URL=ws://localhost:8000/ws/parking" >> .env
```

3. Ejecutar el frontend:
```bash
npm run dev
```

## 📖 Uso

### Configuración Inicial

1. **Configurar Cámara IP**: Editar `CAMERA_URL` en `.env` o `docker-compose.yml`

2. **Configurar Zonas de Estacionamiento**:
   - La configuración por defecto incluye una zona de ejemplo
   - Personalizar en MongoDB colección `configurations`:
   ```javascript
   {
     "camera_id": "cam_001",
     "parking_zones": [
       {
         "zone_id": "zone_1",
         "type": "parallel",
         "baseline": [[100, 300], [700, 300]],
         "width_meters": 2.5
       }
     ]
   }
   ```

3. **Ajustar Parámetros de Detección**:
   - `yolo_confidence`: Umbral de confianza YOLO (default: 0.5)
   - `min_space_length`: Longitud mínima del espacio en metros (default: 4.5)
   - `min_space_width`: Ancho mínimo del espacio en metros (default: 2.2)
   - `temporal_filter_frames`: Frames para filtrado temporal (default: 30)

### API Endpoints

#### Espacios
- `GET /api/spaces` - Listar todos los espacios
- `GET /api/spaces/{space_id}` - Detalle de un espacio
- `GET /api/spaces/{space_id}/history` - Histórico de un espacio

#### Configuración
- `GET /api/config` - Obtener configuración activa
- `POST /api/config/calibration` - Guardar calibración
- `PUT /api/config/params` - Actualizar parámetros

#### Analíticas
- `GET /api/analytics` - Reporte de analíticas
- `GET /api/analytics/occupancy` - Estadísticas de ocupación

#### Sistema
- `GET /api/health` - Health check
- `GET /api/logs` - Logs del sistema

#### WebSocket
- `WS /ws/parking` - Stream de actualizaciones en tiempo real

### Mensajes WebSocket

**Cliente → Servidor:**
```json
{
  "type": "ping",
  "timestamp": "2026-01-02T10:30:00Z"
}
```

**Servidor → Cliente (Estado inicial):**
```json
{
  "type": "initial_state",
  "spaces": [...],
  "timestamp": "2026-01-02T10:30:00Z"
}
```

**Servidor → Cliente (Actualización):**
```json
{
  "type": "parking_update",
  "timestamp": "2026-01-02T10:30:05Z",
  "changes": [
    {
      "space_id": "space_3",
      "old_status": "occupied",
      "new_status": "available",
      "confidence": 0.89
    }
  ]
}
```

## 🏗️ Arquitectura

```
Cámara IP → Backend (FastAPI + YOLO) → WebSocket → Frontend (React)
                ↓
           MongoDB (Base de Datos)
```

### Stack Tecnológico

**Backend:**
- FastAPI - Framework web
- Ultralytics YOLO - Detección de vehículos
- OpenCV - Procesamiento de video
- MongoDB - Base de datos
- Motor - Driver async de MongoDB

**Frontend:**
- React 18 - Framework UI
- Vite - Build tool
- React Konva - Canvas/Visualización
- Zustand - Estado global
- Axios - Cliente HTTP
- Framer Motion - Animaciones
- Tailwind CSS - Estilos

## 📊 Colecciones de MongoDB

- `configurations` - Configuraciones de cámara y zonas
- `parking_spaces` - Estado actual de espacios
- `parking_events` - Histórico de cambios
- `vehicle_detections` - Detecciones de vehículos
- `analytics` - Estadísticas agregadas
- `system_logs` - Logs del sistema

## 🔧 Desarrollo

### Estructura del Proyecto

```
street-parking-analyzer/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── database/
│   │   ├── models/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── config.py
│   │   └── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── stores/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── docker-compose.yml
├── CLAUDE.md
└── README.md
```

### Comandos de MongoDB

```bash
# Conectar a MongoDB
mongosh

# Conectar a MongoDB en Docker
docker exec -it parking_mongodb mongosh -u admin -p password123

# Ver colecciones
use parking_analyzer
show collections

# Query de ejemplo
db.parking_spaces.find({ status: "available" })
```

## 🐛 Solución de Problemas

### El backend no puede conectarse a la cámara
- Verificar que la URL de la cámara sea correcta
- Verificar que la cámara sea accesible desde la red
- Revisar logs: `docker logs parking_backend`

### El frontend no recibe actualizaciones
- Verificar que WebSocket esté conectado (indicador verde)
- Revisar la consola del navegador para errores
- Verificar que el backend esté corriendo

### Error de detección de YOLO
- Verificar que el modelo YOLO se haya descargado correctamente
- Revisar los logs del backend
- Ajustar el umbral de confianza si es necesario

## 📝 Licencia

MIT

## 👥 Autor

Desarrollado siguiendo las especificaciones en CLAUDE.md

## 🙏 Agradecimientos

- Ultralytics por YOLO
- FastAPI
- React y el ecosistema de React
- MongoDB
