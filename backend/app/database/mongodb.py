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
        try:
            # configurations
            await self.db.configurations.create_index([("camera_id", ASCENDING)], unique=True)
            await self.db.configurations.create_index([("active", ASCENDING)])

            # parking_spaces
            await self.db.parking_spaces.create_index([("space_id", ASCENDING)], unique=True)
            await self.db.parking_spaces.create_index([("camera_id", ASCENDING), ("status", ASCENDING)])
            await self.db.parking_spaces.create_index([("last_updated", DESCENDING)])
            await self.db.parking_spaces.create_index([("coordinates", "2dsphere")])

            # parking_events
            await self.db.parking_events.create_index([("space_id", ASCENDING), ("timestamp", DESCENDING)])
            await self.db.parking_events.create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
            await self.db.parking_events.create_index([("timestamp", DESCENDING)])
            await self.db.parking_events.create_index([("timestamp", ASCENDING)], expireAfterSeconds=7776000)  # TTL: 90 días

            # vehicle_detections
            await self.db.vehicle_detections.create_index([("vehicle_id", ASCENDING), ("camera_id", ASCENDING)])
            await self.db.vehicle_detections.create_index([("camera_id", ASCENDING), ("active", ASCENDING)])
            await self.db.vehicle_detections.create_index([("last_seen", DESCENDING)])
            await self.db.vehicle_detections.create_index([("last_seen", ASCENDING)], expireAfterSeconds=86400)  # TTL: 24 horas

            # analytics
            await self.db.analytics.create_index([("camera_id", ASCENDING), ("date", DESCENDING), ("hour", DESCENDING)], unique=True)
            await self.db.analytics.create_index([("date", DESCENDING)])

            # system_logs
            await self.db.system_logs.create_index([("timestamp", DESCENDING)])
            await self.db.system_logs.create_index([("level", ASCENDING), ("timestamp", DESCENDING)])
            await self.db.system_logs.create_index([("timestamp", ASCENDING)], expireAfterSeconds=2592000)  # TTL: 30 días
        except Exception as e:
            print(f"Error creating indexes: {e}")

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
        if "created_at" not in config:
            config["created_at"] = datetime.utcnow()
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
        if "created_at" not in space:
            space["created_at"] = datetime.utcnow()
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
