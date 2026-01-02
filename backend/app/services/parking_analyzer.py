from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


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

    def _analyze_parallel_parking(self, vehicles: List[Dict], zone: Dict) -> List[Dict]:
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
                gap_end = (
                    projected_vehicles[0]["position_on_line"]
                    if projected_vehicles
                    else self._get_zone_length(zone)
                )
            elif i == len(projected_vehicles):
                # Gap después del último vehículo
                gap_start = projected_vehicles[-1]["position_on_line"]
                gap_end = self._get_zone_length(zone)
            else:
                # Gap entre vehículos
                gap_start = projected_vehicles[i - 1]["position_on_line"]
                gap_end = projected_vehicles[i]["position_on_line"]

            gap_length_px = gap_end - gap_start
            gap_length_m = gap_length_px / self.pixels_per_meter

            # Verificar si el gap es suficiente para un espacio
            if gap_length_m >= self.min_space_length:
                space = self._create_space_from_gap(gap_start, gap_end, zone, "available")
                spaces.append(space)

        return spaces

    def _project_vehicles_to_line(
        self, vehicles: List[Dict], baseline: List[List[float]]
    ) -> List[Dict]:
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
            if line_len == 0:
                continue
            line_unitvec = line_vec / line_len

            # Proyección
            vec_to_point = center - line_start
            projection_length = np.dot(vec_to_point, line_unitvec)

            # Solo considerar vehículos cerca de la línea
            projection_point = line_start + projection_length * line_unitvec
            distance_to_line = np.linalg.norm(center - projection_point)

            max_distance_px = 3 * self.pixels_per_meter  # 3 metros
            if distance_to_line < max_distance_px and 0 <= projection_length <= line_len:
                projected.append(
                    {
                        **vehicle,
                        "position_on_line": projection_length,
                        "distance_to_line": distance_to_line,
                    }
                )

        return projected

    def _create_space_from_gap(
        self, gap_start: float, gap_end: float, zone: Dict, status: str
    ) -> Dict:
        """Crear objeto de espacio de estacionamiento"""
        # Calcular polígono del espacio
        baseline = zone["baseline"]
        line_start = np.array(baseline[0])
        line_end = np.array(baseline[1])
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            line_vec = np.array([1, 0])
        else:
            line_vec = line_vec / line_len
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
                "length": (gap_end - gap_start) / self.pixels_per_meter,
            },
            "confidence": 0.85,  # Calcular basado en detecciones
            "last_updated": datetime.utcnow(),
        }

    def _analyze_perpendicular_parking(self, vehicles: List[Dict], zone: Dict) -> List[Dict]:
        """Análisis para estacionamiento perpendicular"""
        # Implementación similar pero proyectando en grid perpendicular
        # TODO: Implementar en fase posterior
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
                space["confidence"] = status_counts[most_common_status] / len(
                    self.space_history[space_id]
                )

            filtered.append(space)

        return filtered

    def _get_zone_length(self, zone: Dict) -> float:
        """Calcular longitud de la zona en píxeles"""
        baseline = zone["baseline"]
        return np.linalg.norm(np.array(baseline[1]) - np.array(baseline[0]))
