"""
Perception System
=================
Soyutlanmış çevre algı sistemi.
Tüm kamera ve sensörleri koordine eder.

Rol dağılımı:
- LaneCamera → EKF lokalizasyon yardımcısı (lateral update)
- SurroundCamera → engel/yaya farkındalığı 
- DoorCamera → kapı güvenliği
- ProximitySensor → durak yanaşma + kapı kör nokta
"""
import numpy as np
from project.sensors.lane_camera import LaneCameraModel
from project.sensors.surround_camera import SurroundCameraModel
from project.sensors.door_camera import DoorCameraModel
from project.sensors.proximity_sensor import ProximitySensorModel


class PerceptionOutput:
    """Bir simülasyon adımı için algı çıktısı."""
    def __init__(self):
        # Şerit algısı
        self.lane_lateral_error = 0.0
        self.lane_heading_error = 0.0
        self.lane_confidence = 1.0
        self.lane_nx = 1.0
        self.lane_ny = 0.0
        self.lane_ref_x = 0.0
        self.lane_ref_y = 0.0
        
        # Yaklaşım sensörü
        self.stop_distance = None
        self.is_near_stop = False
        
        # Kapı güvenliği
        self.front_door_status = "clear"
        self.rear_door_status = "clear"
        self.is_safe_to_close = True
        self.door_positions = {}
        
        # Çevre algı (kamera)
        self.fov_polygons = {}
        self.obstacles_detected = []
        
        # Telemetri
        self.active_cameras = []


class PerceptionSystem:
    """
    Tüm algı sensörlerini koordine eden üst düzey sistem.
    
    Kullanım:
        perception = PerceptionSystem(rng_seed=42)
        output = perception.update(bus_state, route_waypoints, ...)
    """
    
    def __init__(self, rng_seed=42):
        # Sensör alt sistemleri
        self.lane_camera = LaneCameraModel(rng_seed=rng_seed)
        self.surround_camera = SurroundCameraModel()
        self.door_camera = DoorCameraModel(rng_seed=rng_seed)
        self.proximity = ProximitySensorModel(rng_seed=rng_seed)
        
        # Animasyon için kamera görünürlük renkleri
        self.camera_colors = {
            'front': '#00e5ff',           # Cyan — şerit kamerası
            'rear': '#7b1fa2',            # Mor — arka güvenlik
            'left': '#f57f17',            # Turuncu — sol yan
            'right': '#2e7d32',           # Yeşil — sağ yan
            'front_left_blind': '#c62828', # Kırmızı — kör nokta
            'front_right_blind': '#ad1457', # Pembe — kör nokta
        }
        
    def update(self, bus_state, route_waypoints, 
               stop_positions=None, active_passengers=None,
               is_gnss_degraded=False, curvature=0.0,
               is_near_stop=False):
        """
        Tüm sensörleri güncelle ve algı çıktısı üret.
        
        Args:
            bus_state: (x, y, theta, v) otobüs durumu
            route_waypoints: (N, 2) rota noktaları
            stop_positions: [(x, y)] durak global koordinatları
            active_passengers: [(x, y)] aktif yolcu pozisyonları
            is_gnss_degraded: GNSS bozulma durumu
            curvature: Mevcut eğrilik
            is_near_stop: Durağa yakın mı
            
        Returns:
            PerceptionOutput
        """
        output = PerceptionOutput()
        
        if active_passengers is None:
            active_passengers = []
        if stop_positions is None:
            stop_positions = []
        
        # ─── 1. Şerit Kamerası ───────────────────────────────────────────
        (z_lat, z_head, conf, nx, ny, ref_x, ref_y) = \
            self.lane_camera.get_measurement(
                bus_state, route_waypoints,
                is_gnss_degraded=is_gnss_degraded,
                curvature=curvature,
                is_near_stop=is_near_stop
            )
        output.lane_lateral_error = z_lat
        output.lane_heading_error = z_head
        output.lane_confidence = conf
        output.lane_nx = nx
        output.lane_ny = ny
        output.lane_ref_x = ref_x
        output.lane_ref_y = ref_y
        
        # ─── 2. Yaklaşım Sensörü ─────────────────────────────────────────
        for stop_xy in stop_positions:
            z_prox, is_close = self.proximity.get_stop_proximity(bus_state, stop_xy)
            if z_prox is not None:
                output.stop_distance = z_prox
                output.is_near_stop = True
                break
        
        # ─── 3. Kapı Kamerası ────────────────────────────────────────────
        safe, front_status, rear_status = self.door_camera.is_safe_to_close(
            bus_state, active_passengers
        )
        output.front_door_status = front_status
        output.rear_door_status = rear_status
        output.is_safe_to_close = safe
        output.door_positions = self.door_camera.get_all_door_positions(bus_state)
        
        # ─── 4. Çevre Kameraları ─────────────────────────────────────────
        output.fov_polygons = self.surround_camera.get_all_fov_polygons(bus_state)
        
        # Ön kamerayla engel tespiti
        for cam_name in ['front', 'right', 'left']:
            detected, min_dist = self.surround_camera.detect_objects_in_view(
                bus_state, cam_name, active_passengers
            )
            if detected:
                output.obstacles_detected.extend(detected)
                
        output.active_cameras = list(self.surround_camera.cameras.keys())
        
        return output
    
    def get_visualization_data(self, bus_state):
        """
        Animasyon için kapsamlı görselleştirme verisi.
        
        Returns:
            dict: Kamera FOV'ları, kapı pozisyonları, lane camera konisi
        """
        return {
            'camera_fovs': self.surround_camera.get_all_fov_polygons(bus_state),
            'lane_camera_cone': self.lane_camera.get_visualization_data(bus_state),
            'door_zones': self.door_camera.get_visualization_data(bus_state),
            'camera_colors': self.camera_colors,
        }
