import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "parking_analyzer")

    # Camera
    CAMERA_URL: str = os.getenv("CAMERA_URL", "rtsp://192.168.1.100:554/stream")
    CAMERA_ID: str = os.getenv("CAMERA_ID", "cam_001")

    # YOLO
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

    # Processing
    PROCESSING_FPS: int = int(os.getenv("PROCESSING_FPS", "10"))

    # CORS
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
