"""
Proximity Sensor Model
======================
Kısa menzil ultrasonik/radar yaklaşım sensörü.
Durak platformu kenarına yanaşma mesafesini ölçer.
Kapı bölgesinde kör nokta ve engel tespiti yapar.

Gerçek dünya karşılığı: Bosch Ultrasonic Park Assist, Aptiv Short-Range Radar
"""
import numpy as np


class ProximitySensorModel:
    """
    Çok yönlü yaklaşım sensörü.
    
    Otobüs üzerinde 4 yönde sensör:
    - Ön orta (docking/engel)
    - Sağ ön (kapı yanaşma)
    - Sağ arka (kapı yanaşma)
    - Arka orta (manevra güvenliği)
    
    Her sensör: 0.5 – 5.0 m aralık, ±3 cm gürültü
    """
    
    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)
        
        # Sensör parametreleri
        self.min_range = 0.3    # m
        self.max_range = 20.0   # m (durak yaklaşması için uzun menzil)
        self.close_range = 5.0  # m (kapı bölgesi için kısa menzil)
        self.noise_std = 0.03   # m (3 cm)
        
        # Sensör pozisyonları (araç yerel koordinatları)
        # (x_offset, y_offset, heading_offset)
        self.sensor_configs = {
            'front':       (9.27, 0.0,   0.0),           # Ön orta
            'front_right': (6.0,  -1.27, -np.pi/4),      # Sağ ön (kapı)
            'rear_right':  (0.0,  -1.27, -np.pi/4),      # Sağ arka (kapı)
            'rear':        (-3.0, 0.0,   np.pi),          # Arka orta
            'right':       (3.0,  -1.27, -np.pi/2),      # Sağ orta yan
            'left':        (3.0,  +1.27, +np.pi/2),      # Sol orta yan
        }
        
    def _transform_to_global(self, bus_state, local_x, local_y):
        """Araç yerel koordinatından global koordinata dönüşüm."""
        bx, by, theta, _ = bus_state
        gx = bx + local_x * np.cos(theta) - local_y * np.sin(theta)
        gy = by + local_x * np.sin(theta) + local_y * np.cos(theta)
        return gx, gy
        
    def get_stop_proximity(self, state, stop_position_xy):
        """
        Durağa olan mesafe ölçümü.
        Durak pozisyonu global (x, y) koordinatı.
        
        Returns:
            distance (float or None): Mesafe (m), menzil dışındaysa None
            is_close (bool): 5m altında ise True
        """
        x, y, _, _ = state
        dist_true = np.hypot(stop_position_xy[0] - x, stop_position_xy[1] - y)
        
        if dist_true > self.max_range:
            return None, False
            
        z_dist = dist_true + self.rng.normal(0.0, self.noise_std)
        z_dist = max(self.min_range, z_dist)
        is_close = (z_dist <= self.close_range)
        
        return z_dist, is_close
    
    def get_door_zone_proximity(self, state, door_side='right'):
        """
        Kapı bölgesindeki en yakın engel mesafesi.
        
        Returns:
            right_front_dist: Sağ ön sensör mesafesi
            right_rear_dist: Sağ arka sensör mesafesi
        """
        # Sağ ön sensör pozisyonu
        lx_f, ly_f, _ = self.sensor_configs['front_right']
        gx_f, gy_f = self._transform_to_global(state, lx_f, ly_f)
        
        # Sağ arka sensör pozisyonu
        lx_r, ly_r, _ = self.sensor_configs['rear_right']
        gx_r, gy_r = self._transform_to_global(state, lx_r, ly_r)
        
        return (gx_f, gy_f), (gx_r, gy_r)
    
    def detect_obstacle_in_zone(self, sensor_position, passengers, detection_radius=2.5):
        """
        Verilen sensör pozisyonu çevresinde yolcu/engel tespiti.
        
        Args:
            sensor_position: (x, y) global koordinatı
            passengers: Yolcu pozisyonları listesi [(x, y), ...]
            detection_radius: Tespit yarıçapı (m)
            
        Returns:
            detected (bool): Engel var mı
            count (int): Tespit edilen yolcu sayısı
            nearest_dist (float): En yakın mesafe
        """
        sx, sy = sensor_position
        detected = False
        count = 0
        nearest_dist = float('inf')
        
        for px, py in passengers:
            dist = np.hypot(px - sx, py - sy)
            if dist <= detection_radius:
                detected = True
                count += 1
                nearest_dist = min(nearest_dist, dist)
                
        return detected, count, nearest_dist
    
    def get_all_readings(self, state, obstacles=None):
        """
        Tüm sensör okumalarını döndürür (animasyon ve telemetri için).
        
        Returns:
            dict: Her yön için mesafe okuması
        """
        if obstacles is None:
            obstacles = []
            
        readings = {}
        for sensor_name, (lx, ly, _) in self.sensor_configs.items():
            gx, gy = self._transform_to_global(state, lx, ly)
            # Engel yoksa maksimum menzil
            min_dist = self.max_range
            for ox, oy in obstacles:
                d = np.hypot(ox - gx, oy - gy)
                min_dist = min(min_dist, d)
            readings[sensor_name] = min_dist + self.rng.normal(0.0, self.noise_std)
            
        return readings
