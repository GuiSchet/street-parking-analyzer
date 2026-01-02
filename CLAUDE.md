# Street Parking Analyzer - Sistema de Detección de Espacios de Estacionamiento

## 1. VISIÓN GENERAL DEL SISTEMA

### 1.1 Objetivo
Desarrollar una aplicación web en tiempo real que analice video de una cámara IP para detectar espacios de estacionamiento disponibles en una manzana, usando YOLO para detección de vehículos y mostrando resultados en un mapa interactivo.

### 1.2 Arquitectura General
```
Cámara IP → Backend (FastAPI + YOLO) → WebSocket → Frontend (React)
                ↓
           MongoDB (Base de Datos)
```

### 1.3 Stack Tecnológico

#### Backend
- **Framework**: FastAPI
- **ML/Computer Vision**: Ultralytics YOLO (YOLOv8/YOLOv11)
- **Procesamiento de Video**: OpenCV
- **Base de Datos**: MongoDB
- **WebSocket**: FastAPI WebSockets
- **Python**: 3.10+

#### Frontend
- **Framework**: React 18+
- **Build Tool**: Vite
- **Canvas/Visualización**: React Konva
- **Estado Global**: Zustand
- **HTTP Client**: Axios
- **Animaciones**: Framer Motion

#### Base de Datos
- **MongoDB**: Almacenamiento de configuraciones, histórico y logs
- **Motor ODM**: Motor (async) o PyMongo

---

## 2. BACKEND - FastAPI + YOLO + MongoDB

### 2.1 Estructura del Proyecto Backend

```
backend/
├── app/
│   ├── main.py                 # Punto de entrada FastAPI
│   ├── config.py               # Configuraciones
│   ├── database/
│   │   ├── mongodb.py          # Conexión MongoDB
│   │   └── models.py           # Modelos de documentos
│   ├── models/
│   │   ├── detection.py        # Modelos Pydantic
│   │   └── parking_space.py
│   ├── services/
│   │   ├── camera_service.py   # Conexión con cámara IP
│   │   ├── yolo_service.py     # Detección con YOLO
│   │   ├── parking_analyzer.py # Lógica de análisis de espacios
│   │   ├── websocket_service.py
│   │   └── db_service.py       # Servicio de base de datos
│   ├── api/
│   │   ├── routes.py           # Endpoints REST
│   │   └── websocket.py        # Endpoints WebSocket
│   └── utils/
│       ├── geometry.py         # Cálculos geométricos
│       └── calibration.py      # Calibración de perspectiva
├── requirements.txt
├── .env
└── docker-compose.yml          # MongoDB + Backend
```

### 2.2 Integración con MongoDB

#### 2.2.1 Colecciones de MongoDB

```javascript
// Colección: configurations
{
  "_id": ObjectId("..."),
  "camera_id": "cam_001",
  "camera_url": "rtsp://192.168.1.100:554/stream",
  "roi": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "perspective_matrix": [...],
  "pixels_per_meter": 50,
  "parking_zones": [
    {
      "zone_id": "zone_1",
      "type": "parallel",
      "baseline": [[x1,y1], [x2,y2]],
      "width_meters": 2.5
    }
  ],
  "detection_params": {
    "yolo_confidence": 0.5,
    "min_space_length": 4.5,
    "min_space_width": 2.2,
    "stationary_threshold": 3.0,
    "temporal_filter_frames": 30
  },
  "created_at": ISODate("2026-01-02T10:00:00Z"),
  "updated_at": ISODate("2026-01-02T10:00:00Z"),
  "active": true
}

// Colección: parking_spaces
{
  "_id": ObjectId("..."),
  "space_id": "space_1",
  "camera_id": "cam_001",
  "status": "available",  // "available", "occupied", "uncertain"
  "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "coordinates": {
    "lat": -34.603722,
    "lon": -58.381592
  },
  "dimensions": {
    "width": 2.5,
    "length": 5.0
  },
  "confidence": 0.87,
  "zone_id": "zone_1",
  "last_updated": ISODate("2026-01-02T10:30:00Z"),
  "created_at": ISODate("2026-01-02T09:00:00Z")
}

// Colección: parking_events (histórico de cambios)
{
  "_id": ObjectId("..."),
  "space_id": "space_1",
  "camera_id": "cam_001",
  "event_type": "status_change",  // "status_change", "space_created", "space_removed"
  "old_status": "occupied",
  "new_status": "available",
  "timestamp": ISODate("2026-01-02T10:30:00Z"),
  "confidence": 0.87,
  "metadata": {
    "vehicle_id": 123,
    "duration_occupied": 3600  // segundos
  }
}

// Colección: vehicle_detections
{
  "_id": ObjectId("..."),
  "vehicle_id": 123,
  "camera_id": "cam_001",
  "class": "car",
  "bbox": [x1, y1, x2, y2],
  "position": [cx, cy],
  "confidence": 0.89,
  "velocity": [vx, vy],
  "stationary": true,
  "stationary_time": 5.2,
  "space_id": "space_1",  // null si no está estacionado
  "first_seen": ISODate("2026-01-02T10:25:00Z"),
  "last_seen": ISODate("2026-01-02T10:30:00Z"),
  "active": true
}

// Colección: analytics (estadísticas agregadas)
{
  "_id": ObjectId("..."),
  "camera_id": "cam_001",
  "date": ISODate("2026-01-02T00:00:00Z"),
  "hour": 10,
  "total_spaces": 8,
  "avg_available": 3.2,
  "avg_occupied": 4.8,
  "total_vehicles_detected": 45,
  "peak_occupancy_time": "10:30",
  "min_occupancy_time": "03:00",
  "turnover_rate": 12  // vehículos por día
}

// Colección: system_logs
{
  "_id": ObjectId("..."),
  "level": "error",  // "debug", "info", "warning", "error", "critical"
  "component": "camera_service",
  "message": "Camera connection lost",
  "timestamp": ISODate("2026-01-02T10:30:00Z"),
  "metadata": {
    "camera_id": "cam_001",
    "error_code": "RTSP_TIMEOUT"
  }
}
```

#### 2.2.2 Índices MongoDB Recomendados

```javascript
// configurations
db.configurations.createIndex({ "camera_id": 1 }, { unique: true });
db.configurations.createIndex({ "active": 1 });

// parking_spaces
db.parking_spaces.createIndex({ "space_id": 1 }, { unique: true });
db.parking_spaces.createIndex({ "camera_id": 1, "status": 1 });
db.parking_spaces.createIndex({ "last_updated": -1 });
db.parking_spaces.createIndex({ "coordinates": "2dsphere" });  // Para queries geoespaciales

// parking_events
db.parking_events.createIndex({ "space_id": 1, "timestamp": -1 });
db.parking_events.createIndex({ "camera_id": 1, "timestamp": -1 });
db.parking_events.createIndex({ "timestamp": -1 });
db.parking_events.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 7776000 });  // TTL: 90 días

// vehicle_detections
db.vehicle_detections.createIndex({ "vehicle_id": 1, "camera_id": 1 });
db.vehicle_detections.createIndex({ "camera_id": 1, "active": 1 });
db.vehicle_detections.createIndex({ "last_seen": -1 });
db.vehicle_detections.createIndex({ "last_seen": 1 }, { expireAfterSeconds: 86400 });  // TTL: 24 horas

// analytics
db.analytics.createIndex({ "camera_id": 1, "date": -1, "hour": -1 }, { unique: true });
db.analytics.createIndex({ "date": -1 });

// system_logs
db.system_logs.createIndex({ "timestamp": -1 });
db.system_logs.createIndex({ "level": 1, "timestamp": -1 });
db.system_logs.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 2592000 });  // TTL: 30 días
```

#### 2.2.3 Servicio de Base de Datos (db_service.py)

```python
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import os

class DatabaseService:
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.db = None

    async def connect(self):
        """Conectar a MongoDB"""
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[os.getenv("DB_NAME", "parking_analyzer")]
        await self._create_indexes()

    async def disconnect(self):
        """Cerrar conexión"""
        if self.client:
            self.client.close()

    async def _create_indexes(self):
        """Crear índices si no existen"""
        # Implementar creación de índices
        pass

    # === CONFIGURACIONES ===
    async def get_active_config(self, camera_id: str) -> Optional[Dict]:
        """Obtener configuración activa de cámara"""
        return await self.db.configurations.find_one({
            "camera_id": camera_id,
            "active": True
        })

    async def save_config(self, config: Dict) -> str:
        """Guardar/actualizar configuración"""
        config["updated_at"] = datetime.utcnow()
        result = await self.db.configurations.update_one(
            {"camera_id": config["camera_id"]},
            {"$set": config},
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else "updated"

    # === ESPACIOS DE ESTACIONAMIENTO ===
    async def upsert_parking_space(self, space: Dict) -> str:
        """Crear o actualizar espacio de estacionamiento"""
        space["last_updated"] = datetime.utcnow()
        result = await self.db.parking_spaces.update_one(
            {"space_id": space["space_id"]},
            {"$set": space},
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else space["space_id"]

    async def get_spaces_by_camera(self, camera_id: str) -> List[Dict]:
        """Obtener todos los espacios de una cámara"""
        cursor = self.db.parking_spaces.find({"camera_id": camera_id})
        return await cursor.to_list(length=100)

    async def update_space_status(self, space_id: str, new_status: str,
                                 confidence: float) -> bool:
        """Actualizar estado de un espacio"""
        # Primero obtener el estado anterior para crear evento
        old_space = await self.db.parking_spaces.find_one({"space_id": space_id})

        # Actualizar espacio
        result = await self.db.parking_spaces.update_one(
            {"space_id": space_id},
            {
                "$set": {
                    "status": new_status,
                    "confidence": confidence,
                    "last_updated": datetime.utcnow()
                }
            }
        )

        # Registrar evento si hubo cambio de estado
        if old_space and old_space["status"] != new_status:
            await self.log_parking_event(
                space_id=space_id,
                camera_id=old_space["camera_id"],
                old_status=old_space["status"],
                new_status=new_status,
                confidence=confidence
            )

        return result.modified_count > 0

    # === EVENTOS ===
    async def log_parking_event(self, space_id: str, camera_id: str,
                                old_status: str, new_status: str,
                                confidence: float, metadata: Dict = None):
        """Registrar evento de cambio de estado"""
        event = {
            "space_id": space_id,
            "camera_id": camera_id,
            "event_type": "status_change",
            "old_status": old_status,
            "new_status": new_status,
            "timestamp": datetime.utcnow(),
            "confidence": confidence,
            "metadata": metadata or {}
        }
        await self.db.parking_events.insert_one(event)

    async def get_space_history(self, space_id: str, hours: int = 24) -> List[Dict]:
        """Obtener histórico de eventos de un espacio"""
        since = datetime.utcnow() - timedelta(hours=hours)
        cursor = self.db.parking_events.find({
            "space_id": space_id,
            "timestamp": {"$gte": since}
        }).sort("timestamp", DESCENDING)
        return await cursor.to_list(length=1000)

    # === DETECCIONES DE VEHÍCULOS ===
    async def upsert_vehicle_detection(self, detection: Dict) -> str:
        """Guardar/actualizar detección de vehículo"""
        detection["last_seen"] = datetime.utcnow()
        result = await self.db.vehicle_detections.update_one(
            {
                "vehicle_id": detection["vehicle_id"],
                "camera_id": detection["camera_id"]
            },
            {
                "$set": detection,
                "$setOnInsert": {"first_seen": datetime.utcnow()}
            },
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else "updated"

    async def deactivate_stale_vehicles(self, camera_id: str, threshold_seconds: int = 10):
        """Marcar como inactivos vehículos que no se han visto recientemente"""
        threshold = datetime.utcnow() - timedelta(seconds=threshold_seconds)
        await self.db.vehicle_detections.update_many(
            {
                "camera_id": camera_id,
                "last_seen": {"$lt": threshold},
                "active": True
            },
            {"$set": {"active": False}}
        )

    # === ANALÍTICAS ===
    async def save_hourly_analytics(self, camera_id: str, analytics: Dict):
        """Guardar estadísticas por hora"""
        now = datetime.utcnow()
        analytics.update({
            "camera_id": camera_id,
            "date": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "hour": now.hour
        })
        await self.db.analytics.update_one(
            {
                "camera_id": camera_id,
                "date": analytics["date"],
                "hour": analytics["hour"]
            },
            {"$set": analytics},
            upsert=True
        )

    async def get_analytics_report(self, camera_id: str, days: int = 7) -> List[Dict]:
        """Obtener reporte de analíticas"""
        since = datetime.utcnow() - timedelta(days=days)
        cursor = self.db.analytics.find({
            "camera_id": camera_id,
            "date": {"$gte": since}
        }).sort("date", DESCENDING)
        return await cursor.to_list(length=1000)

    # === LOGS ===
    async def log(self, level: str, component: str, message: str, metadata: Dict = None):
        """Registrar log del sistema"""
        log_entry = {
            "level": level,
            "component": component,
            "message": message,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        await self.db.system_logs.insert_one(log_entry)

    async def get_recent_logs(self, level: str = None, limit: int = 100) -> List[Dict]:
        """Obtener logs recientes"""
        query = {"level": level} if level else {}
        cursor = self.db.system_logs.find(query).sort("timestamp", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

# Instancia global
db_service = DatabaseService()
```

### 2.3 Componentes del Backend

#### 2.3.1 Servicio de Cámara (camera_service.py)

```python
import cv2
import asyncio
from typing import Optional
import numpy as np

class CameraService:
    """
    Funcionalidades principales:
    - Conectar a cámara IP vía RTSP/HTTP
    - Capturar frames continuamente
    - Mantener buffer de frames
    - Reconexión automática en caso de fallo
    - Configuración de resolución y FPS
    """

    def __init__(self, camera_url: str, fps: int = 10):
        self.camera_url = camera_url
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    async def connect(self) -> bool:
        """Conectar a la cámara"""
        self.cap = cv2.VideoCapture(self.camera_url)
        if self.cap.isOpened():
            self.reconnect_attempts = 0
            await db_service.log("info", "camera_service",
                                f"Connected to camera: {self.camera_url}")
            return True
        return False

    async def start_capture_loop(self):
        """Loop principal de captura de frames"""
        self.is_running = True
        frame_interval = 1.0 / self.fps

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await db_service.log("warning", "camera_service",
                                        "Camera disconnected, attempting reconnection")
                    self.reconnect_attempts += 1
                    await asyncio.sleep(5)
                    await self.connect()
                    continue
                else:
                    await db_service.log("error", "camera_service",
                                        "Max reconnection attempts reached")
                    break

            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            else:
                await db_service.log("warning", "camera_service",
                                    "Failed to read frame")

            await asyncio.sleep(frame_interval)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Obtener el último frame capturado"""
        return self.latest_frame

    async def stop(self):
        """Detener captura y liberar recursos"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        await db_service.log("info", "camera_service", "Camera service stopped")
```

#### 2.3.2 Servicio YOLO (yolo_service.py)

```python
from ultralytics import YOLO
import numpy as np
from typing import List, Dict

class YOLOService:
    """
    Funcionalidades:
    - Cargar modelo YOLO (YOLOv8 o YOLOv11)
    - Detectar vehículos en frames
    - Tracking de vehículos entre frames
    - Filtrar detecciones por confianza
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck en COCO

    async def detect_vehicles(self, frame: np.ndarray,
                             track: bool = True) -> List[Dict]:
        """
        Detectar vehículos en un frame

        Returns:
            List[Dict]: Lista de detecciones con formato:
            {
                "id": int (si track=True),
                "class": str,
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "center": [cx, cy]
            }
        """
        if track:
            results = self.model.track(frame, persist=True,
                                      conf=self.confidence,
                                      classes=self.vehicle_classes,
                                      verbose=False)
        else:
            results = self.model(frame, conf=self.confidence,
                               classes=self.vehicle_classes,
                               verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detection = {
                    "class": self.model.names[cls],
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                }

                if track and hasattr(box, 'id') and box.id is not None:
                    detection["id"] = int(box.id[0])

                detections.append(detection)

        return detections
```

#### 2.3.3 Analizador de Espacios (parking_analyzer.py)

```python
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict

class ParkingAnalyzer:
    """
    Lógica de Detección de Espacios Libres

    Estrategias:
    1. Análisis de distancias entre vehículos
    2. Grid virtual de espacios
    3. Tracking temporal para reducir falsos positivos
    """

    def __init__(self, config: Dict):
        self.config = config
        self.min_space_length = config["detection_params"]["min_space_length"]
        self.min_space_width = config["detection_params"]["min_space_width"]
        self.pixels_per_meter = config["pixels_per_meter"]
        self.parking_zones = config["parking_zones"]

        # Estado temporal
        self.space_history = defaultdict(list)  # space_id -> [estados]
        self.temporal_filter_frames = config["detection_params"]["temporal_filter_frames"]

    def analyze_spaces(self, vehicles: List[Dict]) -> List[Dict]:
        """
        Analizar espacios disponibles basado en vehículos detectados

        Returns:
            List[Dict]: Espacios de estacionamiento detectados
        """
        spaces = []

        for zone in self.parking_zones:
            if zone["type"] == "parallel":
                zone_spaces = self._analyze_parallel_parking(vehicles, zone)
                spaces.extend(zone_spaces)
            elif zone["type"] == "perpendicular":
                zone_spaces = self._analyze_perpendicular_parking(vehicles, zone)
                spaces.extend(zone_spaces)

        # Aplicar filtro temporal
        filtered_spaces = self._apply_temporal_filter(spaces)

        return filtered_spaces

    def _analyze_parallel_parking(self, vehicles: List[Dict],
                                  zone: Dict) -> List[Dict]:
        """
        Método de análisis de distancias entre vehículos para estacionamiento paralelo
        """
        spaces = []

        # Proyectar vehículos sobre la línea de estacionamiento
        baseline = zone["baseline"]
        projected_vehicles = self._project_vehicles_to_line(vehicles, baseline)

        # Ordenar vehículos por posición en la línea
        projected_vehicles.sort(key=lambda v: v["position_on_line"])

        # Analizar gaps entre vehículos consecutivos
        for i in range(len(projected_vehicles) + 1):
            # Determinar inicio y fin del gap
            if i == 0:
                # Gap antes del primer vehículo
                gap_start = 0  # Inicio de la zona
                gap_end = projected_vehicles[0]["position_on_line"] if projected_vehicles else self._get_zone_length(zone)
            elif i == len(projected_vehicles):
                # Gap después del último vehículo
                gap_start = projected_vehicles[-1]["position_on_line"]
                gap_end = self._get_zone_length(zone)
            else:
                # Gap entre vehículos
                gap_start = projected_vehicles[i-1]["position_on_line"]
                gap_end = projected_vehicles[i]["position_on_line"]

            gap_length_px = gap_end - gap_start
            gap_length_m = gap_length_px / self.pixels_per_meter

            # Verificar si el gap es suficiente para un espacio
            if gap_length_m >= self.min_space_length:
                space = self._create_space_from_gap(
                    gap_start, gap_end, zone, "available"
                )
                spaces.append(space)

        return spaces

    def _project_vehicles_to_line(self, vehicles: List[Dict],
                                  baseline: List[List[float]]) -> List[Dict]:
        """Proyectar vehículos sobre la línea de estacionamiento"""
        projected = []

        for vehicle in vehicles:
            # Calcular proyección del centro del vehículo sobre la baseline
            center = np.array(vehicle["center"])
            line_start = np.array(baseline[0])
            line_end = np.array(baseline[1])

            # Vector de la línea
            line_vec = line_end - line_start
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len

            # Proyección
            vec_to_point = center - line_start
            projection_length = np.dot(vec_to_point, line_unitvec)

            # Solo considerar vehículos cerca de la línea
            projection_point = line_start + projection_length * line_unitvec
            distance_to_line = np.linalg.norm(center - projection_point)

            max_distance_px = 3 * self.pixels_per_meter  # 3 metros
            if distance_to_line < max_distance_px and 0 <= projection_length <= line_len:
                projected.append({
                    **vehicle,
                    "position_on_line": projection_length,
                    "distance_to_line": distance_to_line
                })

        return projected

    def _create_space_from_gap(self, gap_start: float, gap_end: float,
                              zone: Dict, status: str) -> Dict:
        """Crear objeto de espacio de estacionamiento"""
        # Calcular polígono del espacio
        baseline = zone["baseline"]
        line_start = np.array(baseline[0])
        line_end = np.array(baseline[1])
        line_vec = (line_end - line_start) / np.linalg.norm(line_end - line_start)
        perpendicular = np.array([-line_vec[1], line_vec[0]])

        width_px = zone["width_meters"] * self.pixels_per_meter

        # 4 esquinas del polígono
        p1 = line_start + gap_start * line_vec
        p2 = line_start + gap_end * line_vec
        p3 = p2 + width_px * perpendicular
        p4 = p1 + width_px * perpendicular

        polygon = [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()]

        space_id = f"{zone['zone_id']}_space_{int(gap_start)}"

        return {
            "space_id": space_id,
            "zone_id": zone["zone_id"],
            "status": status,
            "polygon": polygon,
            "dimensions": {
                "width": zone["width_meters"],
                "length": (gap_end - gap_start) / self.pixels_per_meter
            },
            "confidence": 0.85,  # Calcular basado en detecciones
            "last_updated": datetime.utcnow()
        }

    def _analyze_perpendicular_parking(self, vehicles: List[Dict],
                                      zone: Dict) -> List[Dict]:
        """Análisis para estacionamiento perpendicular"""
        # Implementación similar pero proyectando en grid perpendicular
        # TODO: Implementar
        return []

    def _apply_temporal_filter(self, spaces: List[Dict]) -> List[Dict]:
        """
        Filtrar espacios usando histórico para reducir falsos positivos
        """
        filtered = []

        for space in spaces:
            space_id = space["space_id"]

            # Agregar al histórico
            self.space_history[space_id].append(space["status"])

            # Mantener solo últimos N frames
            if len(self.space_history[space_id]) > self.temporal_filter_frames:
                self.space_history[space_id].pop(0)

            # Determinar estado más frecuente
            if len(self.space_history[space_id]) >= 5:  # Mínimo 5 frames
                status_counts = {}
                for status in self.space_history[space_id]:
                    status_counts[status] = status_counts.get(status, 0) + 1

                most_common_status = max(status_counts, key=status_counts.get)
                space["status"] = most_common_status
                space["confidence"] = status_counts[most_common_status] / len(self.space_history[space_id])

            filtered.append(space)

        return filtered

    def _get_zone_length(self, zone: Dict) -> float:
        """Calcular longitud de la zona en píxeles"""
        baseline = zone["baseline"]
        return np.linalg.norm(np.array(baseline[1]) - np.array(baseline[0]))
```

#### 2.3.4 Main Application (main.py)

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime

from app.database.mongodb import db_service
from app.services.camera_service import CameraService
from app.services.yolo_service import YOLOService
from app.services.parking_analyzer import ParkingAnalyzer

# Lifespan para inicialización y cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db_service.connect()
    await db_service.log("info", "main", "Application started")

    # Iniciar procesamiento de video en background
    asyncio.create_task(processing_loop())

    yield

    # Shutdown
    await db_service.log("info", "main", "Application shutting down")
    await db_service.disconnect()

app = FastAPI(title="Street Parking Analyzer API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servicios globales
camera_service = None
yolo_service = None
parking_analyzer = None

async def processing_loop():
    """Loop principal de procesamiento"""
    global camera_service, yolo_service, parking_analyzer

    # Cargar configuración desde MongoDB
    config = await db_service.get_active_config("cam_001")
    if not config:
        await db_service.log("error", "main", "No active configuration found")
        return

    # Inicializar servicios
    camera_service = CameraService(config["camera_url"], fps=10)
    await camera_service.connect()
    asyncio.create_task(camera_service.start_capture_loop())

    yolo_service = YOLOService(
        confidence=config["detection_params"]["yolo_confidence"]
    )

    parking_analyzer = ParkingAnalyzer(config)

    # Loop de procesamiento
    while True:
        try:
            # Obtener frame
            frame = camera_service.get_latest_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # Detectar vehículos
            vehicles = await yolo_service.detect_vehicles(frame, track=True)

            # Guardar detecciones en MongoDB
            for vehicle in vehicles:
                await db_service.upsert_vehicle_detection({
                    "vehicle_id": vehicle.get("id"),
                    "camera_id": "cam_001",
                    "class": vehicle["class"],
                    "bbox": vehicle["bbox"],
                    "position": vehicle["center"],
                    "confidence": vehicle["confidence"],
                    "active": True
                })

            # Desactivar vehículos que no se vieron recientemente
            await db_service.deactivate_stale_vehicles("cam_001", threshold_seconds=5)

            # Analizar espacios de estacionamiento
            spaces = parking_analyzer.analyze_spaces(vehicles)

            # Actualizar espacios en MongoDB
            for space in spaces:
                space["camera_id"] = "cam_001"
                await db_service.upsert_parking_space(space)

            # TODO: Broadcast via WebSocket a clientes conectados

            await asyncio.sleep(0.1)  # 10 FPS

        except Exception as e:
            await db_service.log("error", "processing_loop",
                                f"Error in processing loop: {str(e)}")
            await asyncio.sleep(1)

# === ENDPOINTS REST ===

@app.get("/api/spaces")
async def get_parking_spaces(camera_id: str = "cam_001"):
    """Obtener estado actual de todos los espacios"""
    spaces = await db_service.get_spaces_by_camera(camera_id)

    available_count = sum(1 for s in spaces if s["status"] == "available")
    occupied_count = sum(1 for s in spaces if s["status"] == "occupied")

    return {
        "spaces": spaces,
        "total": len(spaces),
        "available": available_count,
        "occupied": occupied_count,
        "timestamp": datetime.utcnow()
    }

@app.get("/api/spaces/{space_id}/history")
async def get_space_history(space_id: str, hours: int = 24):
    """Obtener histórico de un espacio"""
    events = await db_service.get_space_history(space_id, hours)
    return {"space_id": space_id, "events": events}

@app.get("/api/analytics")
async def get_analytics(camera_id: str = "cam_001", days: int = 7):
    """Obtener analíticas agregadas"""
    analytics = await db_service.get_analytics_report(camera_id, days)
    return {"camera_id": camera_id, "analytics": analytics}

@app.post("/api/config/calibration")
async def save_calibration(config: dict):
    """Guardar configuración de calibración"""
    config_id = await db_service.save_config(config)
    return {"status": "success", "config_id": config_id}

@app.get("/api/logs")
async def get_logs(level: str = None, limit: int = 100):
    """Obtener logs del sistema"""
    logs = await db_service.get_recent_logs(level, limit)
    return {"logs": logs}

# === WEBSOCKET ===

active_connections = []

@app.websocket("/ws/parking")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Enviar estado inicial
        spaces = await db_service.get_spaces_by_camera("cam_001")
        await websocket.send_json({
            "type": "initial_state",
            "spaces": spaces
        })

        # Mantener conexión
        while True:
            # Esperar mensajes del cliente (ping/pong)
            data = await websocket.receive_text()

    except Exception as e:
        await db_service.log("warning", "websocket", f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(websocket)

# Función helper para broadcast
async def broadcast_parking_update(changes: List[Dict]):
    """Enviar actualizaciones a todos los clientes WebSocket conectados"""
    message = {
        "type": "parking_update",
        "timestamp": datetime.utcnow().isoformat(),
        "changes": changes
    }

    dead_connections = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            dead_connections.append(connection)

    # Limpiar conexiones muertas
    for conn in dead_connections:
        active_connections.remove(conn)
```

### 2.4 Tecnologías y Dependencias

```txt
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
websockets==12.0
ultralytics==8.1.0
opencv-python==4.9.0
numpy==1.26.3
pydantic==2.5.3
python-multipart==0.0.6
Pillow==10.2.0
scikit-learn==1.4.0
scipy==1.12.0
python-dotenv==1.0.0

# MongoDB
motor==3.3.2                # Async MongoDB driver
pymongo==4.6.1              # MongoDB driver
```

### 2.5 Variables de Entorno

```bash
# backend/.env
MONGODB_URI=mongodb://localhost:27017
DB_NAME=parking_analyzer
CAMERA_URL=rtsp://192.168.1.100:554/stream
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
PROCESSING_FPS=10
CORS_ORIGINS=http://localhost:5173
```

### 2.6 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: parking_mongodb
    restart: always
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password123
      MONGO_INITDB_DATABASE: parking_analyzer
    volumes:
      - mongodb_data:/data/db
      - ./mongo-init:/docker-entrypoint-initdb.d
    networks:
      - parking_network

  backend:
    build: ./backend
    container_name: parking_backend
    restart: always
    ports:
      - "8000:8000"
    environment:
      MONGODB_URI: mongodb://admin:password123@mongodb:27017/parking_analyzer?authSource=admin
      DB_NAME: parking_analyzer
    depends_on:
      - mongodb
    volumes:
      - ./backend:/app
    networks:
      - parking_network

volumes:
  mongodb_data:

networks:
  parking_network:
    driver: bridge
```

---

## 3. FRONTEND - React + Vite

### 3.1 Estructura del Proyecto

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Map/
│   │   │   ├── ParkingMap.jsx
│   │   │   ├── ParkingSpace.jsx
│   │   │   └── MapControls.jsx
│   │   ├── VideoFeed/
│   │   │   └── LiveFeed.jsx
│   │   ├── Dashboard/
│   │   │   ├── Stats.jsx
│   │   │   └── SpaceList.jsx
│   │   └── Config/
│   │       └── Calibration.jsx
│   ├── hooks/
│   │   ├── useWebSocket.js
│   │   └── useParkingData.js
│   ├── services/
│   │   └── api.js
│   ├── stores/
│   │   └── parkingStore.js
│   ├── utils/
│   │   └── coordinates.js
│   ├── App.jsx
│   └── main.jsx
├── package.json
└── vite.config.js
```

### 3.2 Componentes Clave

#### 3.2.1 Hook WebSocket (useWebSocket.js)

```javascript
import { useEffect, useState, useRef } from 'react';

export const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  const ws = useRef(null);
  const reconnectTimeout = useRef(null);

  const connect = () => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setData(message);
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);

      // Reconectar después de 3 segundos
      reconnectTimeout.current = setTimeout(() => {
        console.log('Attempting to reconnect...');
        connect();
      }, 3000);
    };
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url]);

  return { data, connected };
};
```

#### 3.2.2 Store de Estado (parkingStore.js)

```javascript
import { create } from 'zustand';

export const useParkingStore = create((set) => ({
  spaces: [],
  stats: {
    total: 0,
    available: 0,
    occupied: 0
  },

  setSpaces: (spaces) => set({
    spaces,
    stats: {
      total: spaces.length,
      available: spaces.filter(s => s.status === 'available').length,
      occupied: spaces.filter(s => s.status === 'occupied').length
    }
  }),

  updateSpace: (spaceId, updates) => set((state) => ({
    spaces: state.spaces.map(space =>
      space.space_id === spaceId ? { ...space, ...updates } : space
    )
  })),

  applyChanges: (changes) => set((state) => {
    const updatedSpaces = [...state.spaces];
    changes.forEach(change => {
      const index = updatedSpaces.findIndex(s => s.space_id === change.space_id);
      if (index !== -1) {
        updatedSpaces[index] = { ...updatedSpaces[index], status: change.new_status };
      }
    });
    return { spaces: updatedSpaces };
  })
}));
```

#### 3.2.3 Mapa de Estacionamiento (ParkingMap.jsx)

```javascript
import { Stage, Layer, Rect, Line } from 'react-konva';
import ParkingSpace from './ParkingSpace';

const ParkingMap = ({ spaces, width = 800, height = 600 }) => {
  return (
    <Stage width={width} height={height}>
      <Layer>
        {/* Dibujar espacios */}
        {spaces.map(space => (
          <ParkingSpace
            key={space.space_id}
            space={space}
          />
        ))}
      </Layer>
    </Stage>
  );
};

export default ParkingMap;
```

#### 3.2.4 Espacio Individual (ParkingSpace.jsx)

```javascript
import { Line, Text } from 'react-konva';
import { useState } from 'react';

const ParkingSpace = ({ space }) => {
  const [hovered, setHovered] = useState(false);

  const getColor = (status) => {
    switch (status) {
      case 'available': return '#10b981'; // green
      case 'occupied': return '#ef4444';   // red
      case 'uncertain': return '#f59e0b'; // yellow
      default: return '#6b7280';          // gray
    }
  };

  const polygon = space.polygon;
  const flatPoints = polygon.flat();

  return (
    <>
      <Line
        points={flatPoints}
        fill={getColor(space.status)}
        opacity={hovered ? 0.8 : 0.6}
        closed={true}
        stroke="#1f2937"
        strokeWidth={2}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      />
      {hovered && (
        <Text
          x={polygon[0][0]}
          y={polygon[0][1] - 20}
          text={`${space.space_id} (${space.confidence.toFixed(2)})`}
          fontSize={12}
          fill="#ffffff"
        />
      )}
    </>
  );
};

export default ParkingSpace;
```

#### 3.2.5 App Principal (App.jsx)

```javascript
import { useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useParkingStore } from './stores/parkingStore';
import ParkingMap from './components/Map/ParkingMap';
import Stats from './components/Dashboard/Stats';
import axios from 'axios';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/parking';

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
  }, []);

  // Procesar mensajes WebSocket
  useEffect(() => {
    if (!wsData) return;

    if (wsData.type === 'initial_state') {
      setSpaces(wsData.spaces);
    } else if (wsData.type === 'parking_update') {
      applyChanges(wsData.changes);
    }
  }, [wsData]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Street Parking Analyzer</h1>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </header>

        <Stats stats={stats} />

        <div className="mt-8 bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">Live Parking Map</h2>
          <ParkingMap spaces={spaces} />
        </div>
      </div>
    </div>
  );
}

export default App;
```

### 3.3 Dependencias Frontend

```json
{
  "name": "street-parking-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-konva": "^18.2.10",
    "konva": "^9.3.0",
    "axios": "^1.6.5",
    "zustand": "^4.4.7",
    "framer-motion": "^10.18.0",
    "lucide-react": "^0.300.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "vite": "^5.0.8"
  }
}
```

---

## 4. PIPELINE DE PROCESAMIENTO COMPLETO

```
1. Captura Frame (10 FPS)
   ↓
2. Preprocesamiento
   - Crop a ROI (si está configurado)
   - Normalización de imagen
   ↓
3. YOLO Inference
   - Detectar vehículos (car, truck, bus, motorcycle)
   - Filtrar por confianza > 0.5
   - Tracking con IDs persistentes
   ↓
4. Guardar en MongoDB
   - Colección: vehicle_detections
   - Actualizar last_seen
   ↓
5. Análisis de Espacios
   - Proyectar vehículos en zonas de estacionamiento
   - Calcular gaps entre vehículos
   - Identificar espacios disponibles
   ↓
6. Filtrado Temporal
   - Mantener histórico de últimos 30 frames
   - Confirmar cambios persistentes
   ↓
7. Actualizar MongoDB
   - Colección: parking_spaces
   - Registrar eventos en parking_events
   ↓
8. Broadcast WebSocket
   - Enviar solo cambios a clientes conectados
   ↓
9. Render en UI (Frontend)
   - Actualizar mapa en tiempo real
   - Mostrar estadísticas
```

---

## 5. CALIBRACIÓN Y CONFIGURACIÓN INICIAL

### 5.1 Proceso de Setup

1. **Captura de Frame de Referencia**
   - Obtener imagen estática de la cámara
   - Mostrar en interfaz de calibración

2. **Definir ROI (Región de Interés)**
   - Usuario dibuja polígono en la imagen
   - Guardar coordenadas en MongoDB

3. **Configurar Zonas de Estacionamiento**
   - Dibujar líneas base para cada zona
   - Especificar tipo (paralelo/perpendicular)
   - Definir ancho de zona

4. **Calibración de Escala**
   - Marcar objeto de dimensión conocida
   - Calcular píxeles por metro

5. **Ajustar Parámetros de Detección**
   - Umbral de confianza YOLO
   - Dimensiones mínimas de espacio
   - Sensibilidad temporal

6. **Guardar Configuración**
   - Almacenar en colección `configurations`
   - Marcar como activa

---

## 6. COMANDOS DE INICIO

### 6.1 Con Docker

```bash
# Iniciar MongoDB y Backend
docker-compose up -d

# Ver logs
docker-compose logs -f backend
```

### 6.2 Desarrollo Local

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm run dev  # Puerto 5173
```

### 6.3 MongoDB

```bash
# Conectar a MongoDB local
mongosh

# Conectar a MongoDB en Docker
docker exec -it parking_mongodb mongosh -u admin -p password123

# Ver colecciones
use parking_analyzer
show collections

# Query de ejemplo
db.parking_spaces.find({ status: "available" })
```

---

## 7. OPTIMIZACIONES Y CONSIDERACIONES

### 7.1 Performance

- Procesar frames a 5-10 FPS (suficiente para detección)
- Usar modelo YOLO ligero (YOLOv8n o YOLOv11n)
- Implementar cache de detecciones estacionadas
- Índices MongoDB para queries rápidas
- Proyecciones limitadas en queries (solo campos necesarios)

### 7.2 Robustez

- Reconexión automática a cámara
- Manejo de errores con try/except
- Logging estructurado en MongoDB
- Health check endpoints
- TTL indexes para limpiar datos antiguos

### 7.3 Escalabilidad

- Arquitectura preparada para múltiples cámaras
- Sharding de MongoDB por camera_id
- Load balancing de WebSockets
- Cache con Redis (futuro)
- Message queue para procesamiento (Celery/RabbitMQ)

---

## 8. ORDEN DE IMPLEMENTACIÓN SUGERIDO

### Fase 1: MVP Básico (1-2 semanas)
1. Setup FastAPI con endpoints básicos
2. Conectar a MongoDB
3. Implementar captura de cámara IP
4. Integrar YOLO para detección
5. Frontend básico mostrando detecciones

### Fase 2: Análisis de Espacios (1 semana)
1. Implementar calibración de ROI
2. Lógica de detección de espacios (método de distancias)
3. Guardar espacios en MongoDB
4. Mapa simple en frontend

### Fase 3: Tiempo Real (1 semana)
1. WebSocket bidireccional
2. Actualización en tiempo real del mapa
3. Tracking persistente de vehículos
4. Registro de eventos históricos

### Fase 4: Refinamiento (1-2 semanas)
1. Filtrado temporal robusto
2. Analíticas y reportes
3. UI/UX polish
4. Testing y ajuste de parámetros
5. Documentación

---

## 9. ENDPOINTS API COMPLETOS

```python
# Espacios
GET    /api/spaces                    # Listar espacios actuales
GET    /api/spaces/{space_id}         # Detalle de espacio
GET    /api/spaces/{space_id}/history # Histórico de espacio

# Configuración
GET    /api/config                    # Configuración activa
POST   /api/config/calibration        # Guardar calibración
PUT    /api/config/params             # Actualizar parámetros

# Analíticas
GET    /api/analytics                 # Reporte de analíticas
GET    /api/analytics/occupancy       # Gráfico de ocupación

# Vehículos
GET    /api/vehicles/active           # Vehículos detectados actualmente

# Sistema
GET    /api/health                    # Health check
GET    /api/logs                      # Logs del sistema
GET    /api/stream/snapshot           # Frame actual con detecciones

# WebSocket
WS     /ws/parking                    # Stream de actualizaciones
```

---

## 10. ESTRUCTURA DE MENSAJES WEBSOCKET

```javascript
// Cliente → Servidor
{
  "type": "ping",
  "timestamp": "2026-01-02T10:30:00Z"
}

// Servidor → Cliente (Estado inicial)
{
  "type": "initial_state",
  "spaces": [...],
  "timestamp": "2026-01-02T10:30:00Z"
}

// Servidor → Cliente (Actualización)
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

// Servidor → Cliente (Error)
{
  "type": "error",
  "message": "Camera connection lost",
  "timestamp": "2026-01-02T10:30:00Z"
}
```

---

## 11. NOTAS IMPORTANTES

### 11.1 Desafíos Técnicos

1. **Espacios No Delimitados**: Usar clustering y heurísticas de distancia
2. **Cambios de Iluminación**: Normalización y ajuste de parámetros dinámico
3. **Oclusiones**: Tracking persistente con Kalman filter
4. **Vehículos en Movimiento vs Estacionados**: Cálculo de velocidad y tiempo de permanencia

### 11.2 Seguridad

- Autenticación para endpoints administrativos (futuro)
- Rate limiting en API
- Validación de inputs con Pydantic
- CORS configurado correctamente
- MongoDB con autenticación en producción

### 11.3 Monitoreo

- Logs estructurados en MongoDB con TTL
- Métricas de performance (FPS procesamiento)
- Alertas de desconexión de cámara
- Dashboard de estadísticas

---

## 12. RECURSOS Y REFERENCIAS

- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **MongoDB Motor**: https://motor.readthedocs.io/
- **React Konva**: https://konvajs.org/docs/react/
- **OpenCV**: https://docs.opencv.org/

---

**Versión**: 1.0
**Última Actualización**: 2026-01-02
**Autor**: Claude AI
**Licencia**: MIT
