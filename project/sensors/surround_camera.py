"""
Surround Camera System
======================
Otonom otobüs çevre kamera sistemi.
6 kamera ile 360° çevre algısını soyutlanmış model olarak simüle eder.

Gerçek dünya karşılığı: Mobileye 8-kamera sistemi, Tesla Vision, Waymo kamera halkalığı
"""
import numpy as np


class CameraConfig:
    """Tek kamera konfigürasyonu."""
    def __init__(self, name, local_x, local_y, heading_offset_deg, fov_deg, range_m):
        self.name = name
        self.local_x = local_x
        self.local_y = local_y
        self.heading_offset = np.deg2rad(heading_offset_deg)
        self.fov = np.deg2rad(fov_deg)
        self.range_m = range_m


class SurroundCameraModel:
    """
    Otobüs etrafındaki 6 kameranın soyutlanmış algı modeli.
    
    Kamera yerleşimi (araç yerel koordinatları):
    ┌─────────────────────────────────┐
    │  [Ön Kamera]                    │ → (x=9.27, y=0) 
    │  [Sol Yan] [Sağ Yan]            │ → (x=3, y=±1.27)
    │  [Sol KN]  [Sağ KN]  [Arka]    │ → kör nokta + arka
    └─────────────────────────────────┘
    """
    
    def __init__(self):
        self.cameras = {
            'front': CameraConfig('front', 9.27, 0.0, 0, 70, 40.0),
            'rear': CameraConfig('rear', -3.0, 0.0, 180, 70, 20.0),
            'left': CameraConfig('left', 3.0, 1.27, 90, 120, 15.0),
            'right': CameraConfig('right', 3.0, -1.27, -90, 120, 15.0),
            'front_left_blind': CameraConfig('front_left_blind', 7.0, 1.27, 45, 90, 10.0),
            'front_right_blind': CameraConfig('front_right_blind', 7.0, -1.27, -45, 90, 10.0),
        }
        
        # Algı görevleri
        self.lane_cameras = ['front']              # Şerit algısı
        self.obstacle_cameras = list(self.cameras.keys())  # Engel tespiti
        self.blindspot_cameras = ['front_left_blind', 'front_right_blind']  # Kör nokta
        
    def get_camera_fov_polygon(self, bus_state, camera_name):
        """
        Belirtilen kameranın görüş alanı polygon noktaları.
        Animasyon için koni şeklinde çizim verisi.
        
        Returns:
            list of (x, y): FOV polygon köşe noktaları
        """
        bx, by, bus_theta, _ = bus_state
        cam = self.cameras[camera_name]
        
        # Kamera global pozisyonu
        cam_gx = bx + cam.local_x * np.cos(bus_theta) - cam.local_y * np.sin(bus_theta)
        cam_gy = by + cam.local_x * np.sin(bus_theta) + cam.local_y * np.cos(bus_theta)
        
        # Kamera global yönelimi
        cam_global_heading = bus_theta + cam.heading_offset
        
        # Koni polygon oluştur
        n_arc = 10
        angles = np.linspace(
            cam_global_heading - cam.fov / 2,
            cam_global_heading + cam.fov / 2,
            n_arc
        )
        
        polygon = [(cam_gx, cam_gy)]
        for a in angles:
            px = cam_gx + cam.range_m * np.cos(a)
            py = cam_gy + cam.range_m * np.sin(a)
            polygon.append((px, py))
        polygon.append((cam_gx, cam_gy))
        
        return polygon
    
    def get_all_fov_polygons(self, bus_state):
        """Tüm kameraların FOV polygon verisi."""
        return {
            name: self.get_camera_fov_polygon(bus_state, name)
            for name in self.cameras
        }
    
    def detect_objects_in_view(self, bus_state, camera_name, objects):
        """
        Verilen kameranın görüş alanında nesne tespiti.
        
        Args:
            bus_state: Otobüs durumu (x, y, theta, v)
            camera_name: Kamera adı
            objects: [(x, y), ...] nesne listesi
            
        Returns:
            detected_objects: Görüş alanındaki nesneler
            min_distance: En yakın nesne mesafesi
        """
        bx, by, bus_theta, _ = bus_state
        cam = self.cameras[camera_name]
        
        cam_gx = bx + cam.local_x * np.cos(bus_theta) - cam.local_y * np.sin(bus_theta)
        cam_gy = by + cam.local_x * np.sin(bus_theta) + cam.local_y * np.cos(bus_theta)
        cam_global_heading = bus_theta + cam.heading_offset
        
        detected = []
        min_dist = float('inf')
        
        for obj_x, obj_y in objects:
            dx = obj_x - cam_gx
            dy = obj_y - cam_gy
            dist = np.hypot(dx, dy)
            
            if dist > cam.range_m:
                continue
                
            angle_to_obj = np.arctan2(dy, dx)
            bearing = (angle_to_obj - cam_global_heading + np.pi) % (2 * np.pi) - np.pi
            
            if abs(bearing) <= cam.fov / 2:
                detected.append((obj_x, obj_y, dist))
                min_dist = min(min_dist, dist)
                
        return detected, min_dist
    
    def get_lane_visibility(self, bus_state, is_gnss_degraded=False, curvature=0.0):
        """
        Ön kameranın şerit çizgisi görünürlüğünü değerlendirir.
        
        Returns:
            lane_visible (bool): Şerit görünüyor mu
            confidence (float): Güven değeri [0, 1]
        """
        base_conf = 0.92
        if is_gnss_degraded:
            base_conf -= 0.15
        if abs(curvature) > 0.015:
            base_conf -= 0.25
            
        confidence = float(np.clip(base_conf, 0.1, 1.0))
        lane_visible = confidence > 0.3
        
        return lane_visible, confidence
