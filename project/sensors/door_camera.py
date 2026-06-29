"""
Door Camera Model
=================
Kapı güvenlik kamerası — yolcu varlığı tespiti.
Her kapıda 1 adet kamera. Kapı bölgesinde (2m yarıçap) yolcu/engel algılar.

Gerçek dünya karşılığı: Iris Dynamics kapı güvenlik sistemi, 
INIT AI kapı yolcu analitik sistemi
"""
import numpy as np


class DoorCameraModel:
    """
    Sağ taraf kapı güvenlik kamerası sistemi.
    
    Otobüste 2 kapı:
    - Ön kapı: x_local = 7.0 m, y_local = -1.27 m  
    - Arka kapı: x_local = 1.5 m, y_local = -1.27 m
    
    Her kamera:
    - Algı yarıçapı: 2.0 m
    - Yolcu sayma: basit proximity tabanlı
    - Güvenlik durumu: CLEAR / OCCUPIED / WARNING
    """
    
    CLEAR = "clear"
    OCCUPIED = "occupied"
    WARNING = "warning"
    
    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)
        
        # Kapı yerel koordinatları (araç koordinat sistemi)
        self.door_configs = {
            'front_door': {
                'local_x': 7.0,
                'local_y': -1.27,
                'detection_radius': 2.0,   # m
                'safety_radius': 0.8,       # m (çok yakın uyarı)
            },
            'rear_door': {
                'local_x': 1.5,
                'local_y': -1.27,
                'detection_radius': 2.0,
                'safety_radius': 0.8,
            }
        }
        
        # Tespit gürültüsü
        self.false_detection_prob = 0.02  # %2 yanlış pozitif
        self.miss_detection_prob = 0.01   # %1 yanlış negatif
        
    def get_door_position_global(self, bus_state, door_name):
        """
        Kapının global koordinatını hesaplar.
        
        Returns:
            (gx, gy): Global koordinat
        """
        bx, by, theta, _ = bus_state
        cfg = self.door_configs[door_name]
        lx, ly = cfg['local_x'], cfg['local_y']
        
        gx = bx + lx * np.cos(theta) - ly * np.sin(theta)
        gy = by + lx * np.sin(theta) + ly * np.cos(theta)
        
        return gx, gy
    
    def get_all_door_positions(self, bus_state):
        """Tüm kapıların global koordinatları."""
        return {
            name: self.get_door_position_global(bus_state, name)
            for name in self.door_configs
        }
    
    def check_door_zone(self, bus_state, door_name, passenger_positions):
        """
        Belirtilen kapı bölgesinde yolcu tespiti.
        
        Args:
            bus_state: Otobüs durumu
            door_name: 'front_door' veya 'rear_door'
            passenger_positions: [(x, y), ...] yolcu listesi
            
        Returns:
            status: CLEAR / OCCUPIED / WARNING
            passenger_count: Bölgedeki yolcu sayısı
            nearest_distance: En yakın yolcu mesafesi
            door_gx, door_gy: Kapı global konumu
        """
        door_gx, door_gy = self.get_door_position_global(bus_state, door_name)
        cfg = self.door_configs[door_name]
        
        detection_r = cfg['detection_radius']
        safety_r = cfg['safety_radius']
        
        passenger_count = 0
        nearest_dist = float('inf')
        is_warning = False
        
        for px, py in passenger_positions:
            dist = np.hypot(px - door_gx, py - door_gy)
            
            if dist <= detection_r:
                # Gürültü modellemesi
                if self.rng.random() < self.miss_detection_prob:
                    continue  # Yanlış negatif
                passenger_count += 1
                nearest_dist = min(nearest_dist, dist)
                
                if dist <= safety_r:
                    is_warning = True
            
        # Yanlış pozitif kontrolü
        if passenger_count == 0 and self.rng.random() < self.false_detection_prob:
            passenger_count = 1
            nearest_dist = detection_r * 0.9
            
        if is_warning:
            status = self.WARNING
        elif passenger_count > 0:
            status = self.OCCUPIED
        else:
            status = self.CLEAR
            
        return status, passenger_count, nearest_dist, door_gx, door_gy
    
    def is_safe_to_close(self, bus_state, passenger_positions):
        """
        Her iki kapının da kapatılabilir durumda olup olmadığını kontrol eder.
        
        Returns:
            safe (bool): İki kapı da boşsa True
            front_status: Ön kapı durumu
            rear_status: Arka kapı durumu
        """
        front_status, front_count, _, _, _ = self.check_door_zone(
            bus_state, 'front_door', passenger_positions
        )
        rear_status, rear_count, _, _, _ = self.check_door_zone(
            bus_state, 'rear_door', passenger_positions
        )
        
        safe = (front_status == self.CLEAR) and (rear_status == self.CLEAR)
        return safe, front_status, rear_status
    
    def get_visualization_data(self, bus_state):
        """
        Animasyon için kapı pozisyonu ve algı yarıçapı verisi.
        
        Returns:
            dict: Her kapı için (gx, gy, radius) 
        """
        viz = {}
        for door_name, cfg in self.door_configs.items():
            gx, gy = self.get_door_position_global(bus_state, door_name)
            viz[door_name] = {
                'position': (gx, gy),
                'detection_radius': cfg['detection_radius'],
                'safety_radius': cfg['safety_radius'],
            }
        return viz
