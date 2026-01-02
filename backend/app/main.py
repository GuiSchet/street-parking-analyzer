from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from typing import List

from app.database.mongodb import db_service
from app.services.camera_service import CameraService
from app.services.yolo_service import YOLOService
from app.services.parking_analyzer import ParkingAnalyzer
from app.config import settings

# Servicios globales
camera_service = None
yolo_service = None
parking_analyzer = None
active_connections: List[WebSocket] = []


async def broadcast_parking_update(changes: List[dict]):
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
        if conn in active_connections:
            active_connections.remove(conn)


async def processing_loop():
    """Loop principal de procesamiento"""
    global camera_service, yolo_service, parking_analyzer

    # Cargar configuración desde MongoDB
    config = await db_service.get_active_config(settings.CAMERA_ID)
    if not config:
        await db_service.log("error", "main", "No active configuration found")
        # Crear configuración por defecto
        default_config = {
            "camera_id": settings.CAMERA_ID,
            "camera_url": settings.CAMERA_URL,
            "roi": None,
            "perspective_matrix": None,
            "pixels_per_meter": 50.0,
            "parking_zones": [
                {
                    "zone_id": "zone_1",
                    "type": "parallel",
                    "baseline": [[100, 300], [700, 300]],
                    "width_meters": 2.5
                }
            ],
            "detection_params": {
                "yolo_confidence": settings.CONFIDENCE_THRESHOLD,
                "min_space_length": 4.5,
                "min_space_width": 2.2,
                "stationary_threshold": 3.0,
                "temporal_filter_frames": 30
            },
            "active": True
        }
        await db_service.save_config(default_config)
        config = default_config

    # Inicializar servicios
    camera_service = CameraService(config["camera_url"], fps=settings.PROCESSING_FPS)
    await camera_service.connect()
    asyncio.create_task(camera_service.start_capture_loop())

    yolo_service = YOLOService(
        model_path=settings.YOLO_MODEL,
        confidence=config["detection_params"]["yolo_confidence"]
    )

    parking_analyzer = ParkingAnalyzer(config)

    # Loop de procesamiento
    previous_spaces = {}
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
                    "vehicle_id": vehicle.get("id", 0),
                    "camera_id": settings.CAMERA_ID,
                    "class": vehicle["class"],
                    "bbox": vehicle["bbox"],
                    "position": vehicle["center"],
                    "confidence": vehicle["confidence"],
                    "active": True
                })

            # Desactivar vehículos que no se vieron recientemente
            await db_service.deactivate_stale_vehicles(settings.CAMERA_ID, threshold_seconds=5)

            # Analizar espacios de estacionamiento
            spaces = parking_analyzer.analyze_spaces(vehicles)

            # Actualizar espacios en MongoDB y detectar cambios
            changes = []
            for space in spaces:
                space["camera_id"] = settings.CAMERA_ID
                space_id = space["space_id"]

                # Verificar si hubo cambio de estado
                if space_id in previous_spaces:
                    if previous_spaces[space_id]["status"] != space["status"]:
                        changes.append({
                            "space_id": space_id,
                            "old_status": previous_spaces[space_id]["status"],
                            "new_status": space["status"],
                            "confidence": space["confidence"]
                        })

                await db_service.upsert_parking_space(space)
                previous_spaces[space_id] = space

            # Broadcast cambios via WebSocket
            if changes and active_connections:
                await broadcast_parking_update(changes)

            await asyncio.sleep(0.1)  # 10 FPS

        except Exception as e:
            await db_service.log("error", "processing_loop",
                                f"Error in processing loop: {str(e)}")
            await asyncio.sleep(1)


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
    if camera_service:
        await camera_service.stop()
    await db_service.disconnect()


app = FastAPI(title="Street Parking Analyzer API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === ENDPOINTS REST ===

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Street Parking Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "camera_connected": camera_service is not None and camera_service.cap is not None and camera_service.cap.isOpened() if camera_service else False
    }


@app.get("/api/spaces")
async def get_parking_spaces(camera_id: str = settings.CAMERA_ID):
    """Obtener estado actual de todos los espacios"""
    spaces = await db_service.get_spaces_by_camera(camera_id)

    available_count = sum(1 for s in spaces if s.get("status") == "available")
    occupied_count = sum(1 for s in spaces if s.get("status") == "occupied")

    return {
        "spaces": spaces,
        "total": len(spaces),
        "available": available_count,
        "occupied": occupied_count,
        "timestamp": datetime.utcnow()
    }


@app.get("/api/spaces/{space_id}")
async def get_space_detail(space_id: str):
    """Obtener detalle de un espacio específico"""
    space = await db_service.db.parking_spaces.find_one({"space_id": space_id})
    if not space:
        return {"error": "Space not found"}, 404
    return space


@app.get("/api/spaces/{space_id}/history")
async def get_space_history(space_id: str, hours: int = 24):
    """Obtener histórico de un espacio"""
    events = await db_service.get_space_history(space_id, hours)
    return {"space_id": space_id, "events": events}


@app.get("/api/analytics")
async def get_analytics(camera_id: str = settings.CAMERA_ID, days: int = 7):
    """Obtener analíticas agregadas"""
    analytics = await db_service.get_analytics_report(camera_id, days)
    return {"camera_id": camera_id, "analytics": analytics}


@app.get("/api/analytics/occupancy")
async def get_occupancy():
    """Obtener estadísticas de ocupación actuales"""
    spaces = await db_service.get_spaces_by_camera(settings.CAMERA_ID)

    return {
        "total_spaces": len(spaces),
        "available": sum(1 for s in spaces if s.get("status") == "available"),
        "occupied": sum(1 for s in spaces if s.get("status") == "occupied"),
        "uncertain": sum(1 for s in spaces if s.get("status") == "uncertain"),
        "occupancy_rate": (sum(1 for s in spaces if s.get("status") == "occupied") / len(spaces) * 100) if spaces else 0,
        "timestamp": datetime.utcnow()
    }


@app.post("/api/config/calibration")
async def save_calibration(config: dict):
    """Guardar configuración de calibración"""
    config_id = await db_service.save_config(config)
    return {"status": "success", "config_id": config_id}


@app.get("/api/config")
async def get_config(camera_id: str = settings.CAMERA_ID):
    """Obtener configuración activa"""
    config = await db_service.get_active_config(camera_id)
    if not config:
        return {"error": "No configuration found"}, 404
    return config


@app.put("/api/config/params")
async def update_params(params: dict, camera_id: str = settings.CAMERA_ID):
    """Actualizar parámetros de detección"""
    config = await db_service.get_active_config(camera_id)
    if not config:
        return {"error": "No configuration found"}, 404

    config["detection_params"].update(params)
    await db_service.save_config(config)
    return {"status": "success", "params": config["detection_params"]}


@app.get("/api/logs")
async def get_logs(level: str = None, limit: int = 100):
    """Obtener logs del sistema"""
    logs = await db_service.get_recent_logs(level, limit)
    return {"logs": logs}


@app.get("/api/vehicles/active")
async def get_active_vehicles(camera_id: str = settings.CAMERA_ID):
    """Obtener vehículos detectados actualmente"""
    cursor = db_service.db.vehicle_detections.find({
        "camera_id": camera_id,
        "active": True
    })
    vehicles = await cursor.to_list(length=100)
    return {"vehicles": vehicles, "count": len(vehicles)}


# === WEBSOCKET ===

@app.websocket("/ws/parking")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Enviar estado inicial
        spaces = await db_service.get_spaces_by_camera(settings.CAMERA_ID)
        await websocket.send_json({
            "type": "initial_state",
            "spaces": spaces,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Mantener conexión
        while True:
            # Esperar mensajes del cliente (ping/pong)
            data = await websocket.receive_text()

    except WebSocketDisconnect:
        await db_service.log("info", "websocket", "Client disconnected")
    except Exception as e:
        await db_service.log("warning", "websocket", f"WebSocket error: {str(e)}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
