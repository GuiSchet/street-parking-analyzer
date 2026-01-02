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
