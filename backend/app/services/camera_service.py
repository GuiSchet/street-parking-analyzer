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
        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            if self.cap.isOpened():
                self.reconnect_attempts = 0
                print(f"Connected to camera: {self.camera_url}")
                return True
            return False
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False

    async def start_capture_loop(self):
        """Loop principal de captura de frames"""
        from app.database.mongodb import db_service

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
        from app.database.mongodb import db_service

        self.is_running = False
        if self.cap:
            self.cap.release()
        await db_service.log("info", "camera_service", "Camera service stopped")
