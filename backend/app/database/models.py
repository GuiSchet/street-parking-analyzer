from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ParkingStatus(str, Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    UNCERTAIN = "uncertain"


class ParkingType(str, Enum):
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"


class EventType(str, Enum):
    STATUS_CHANGE = "status_change"
    SPACE_CREATED = "space_created"
    SPACE_REMOVED = "space_removed"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Coordinates(BaseModel):
    lat: float
    lon: float


class Dimensions(BaseModel):
    width: float
    length: float


class DetectionParams(BaseModel):
    yolo_confidence: float = 0.5
    min_space_length: float = 4.5
    min_space_width: float = 2.2
    stationary_threshold: float = 3.0
    temporal_filter_frames: int = 30


class ParkingZone(BaseModel):
    zone_id: str
    type: ParkingType
    baseline: List[List[float]]
    width_meters: float


class CameraConfiguration(BaseModel):
    camera_id: str
    camera_url: str
    roi: Optional[List[List[float]]] = None
    perspective_matrix: Optional[List[float]] = None
    pixels_per_meter: float = 50.0
    parking_zones: List[ParkingZone]
    detection_params: DetectionParams = DetectionParams()
    active: bool = True


class ParkingSpace(BaseModel):
    space_id: str
    camera_id: str
    status: ParkingStatus
    polygon: List[List[float]]
    coordinates: Optional[Coordinates] = None
    dimensions: Dimensions
    confidence: float
    zone_id: str
    last_updated: Optional[datetime] = None


class VehicleDetection(BaseModel):
    vehicle_id: int
    camera_id: str
    class_name: str = Field(alias="class")
    bbox: List[float]
    position: List[float]
    confidence: float
    velocity: Optional[List[float]] = None
    stationary: bool = False
    stationary_time: float = 0.0
    space_id: Optional[str] = None
    active: bool = True


class ParkingEvent(BaseModel):
    space_id: str
    camera_id: str
    event_type: EventType
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    timestamp: datetime
    confidence: float
    metadata: Optional[Dict] = {}


class SystemLog(BaseModel):
    level: LogLevel
    component: str
    message: str
    timestamp: datetime
    metadata: Optional[Dict] = {}
