"""
Lane Camera Sensor Model
========================
Ön kamera tabanlı yol çizgisi algılama sistemi.
Şerit merkezine yanal mesafe, yönelim farkı ve çizgi güveni sağlar.

Gerçek dünya karşılığı: Mobileye, Intel RealSense, Bosch Flexray kamera sistemleri
"""
import numpy as np


class LaneCameraModel:
    """
    Ön kamera tabanlı şerit algılama sensörü.
    
    Fiziksel parametreler:
    - FOV: 60° yatay açı
    - Menzil: 30 m ön görüş
    - Çözünürlük: 2 cm lateral (eşdeğer)
    - Çizgi görünürlük güveni: 0.0 - 1.0
    
    Bozulma koşulları:
    - GNSS bozulma bölgesi: güven %20 azalır
    - Yüksek eğrilik (viraj): güven %30 azalır
    - Yakın durak bölgesi: %10 azalır (platform çizgileri karışabilir)
    """
    
    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)
        
        # Sensör parametreleri
        self.fov_deg = 60.0          # yatay görüş açısı
        self.range_m = 30.0          # ön görüş menzili
        self.lateral_noise_std = 0.02  # m (2 cm std dev)
        self.heading_noise_std = np.deg2rad(0.5)  # 0.5 derece
        
        # Güven modeli bileşenleri
        self.base_confidence = 0.95  # Nominal koşulda güven
        
    def get_measurement(self, state, route_waypoints, 
                        is_gnss_degraded=False, curvature=0.0, is_near_stop=False):
        """
        Şerit kamera ölçümü.
        
        Returns:
            lateral_error (float): Şerit merkezinden yanal sapma (m)
            heading_error (float): Yönelimdeki sapma (rad)
            confidence (float): Çizgi görünürlük güveni [0, 1]
            nx, ny (float): Şerit normal vektörü
            ref_x, ref_y (float): Referans noktası koordinatları
        """
        x, y, theta, _ = state
        
        # En yakın waypoint'i bul
        dists = np.hypot(route_waypoints[:, 0] - x, route_waypoints[:, 1] - y)
        idx = np.argmin(dists)
        closest_pt = route_waypoints[idx]
        
        # Teğet vektörü hesapla
        if idx < len(route_waypoints) - 1:
            tx = route_waypoints[idx+1, 0] - closest_pt[0]
            ty = route_waypoints[idx+1, 1] - closest_pt[1]
        else:
            tx = closest_pt[0] - route_waypoints[idx-1, 0]
            ty = closest_pt[1] - route_waypoints[idx-1, 1]
            
        t_len = np.hypot(tx, ty)
        if t_len > 1e-6:
            tx, ty = tx / t_len, ty / t_len
        else:
            tx, ty = 1.0, 0.0
            
        # Normal vektör (sol taraf)
        nx = -ty
        ny = tx
        
        # Gerçek yanal hata
        dx = x - closest_pt[0]
        dy = y - closest_pt[1]
        lateral_error_true = dx * nx + dy * ny
        
        # Gerçek yönelim hatası
        route_heading = np.arctan2(ty, tx)
        heading_error_true = (theta - route_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Güven hesaplama
        confidence = self.base_confidence
        if is_gnss_degraded:
            confidence -= 0.20
        if abs(curvature) > 0.015:   # Sıkı viraj
            confidence -= 0.30
        elif abs(curvature) > 0.005:  # Hafif viraj
            confidence -= 0.10
        if is_near_stop:
            confidence -= 0.10
        confidence = float(np.clip(confidence, 0.1, 1.0))
        
        # Gürültü — güvene bağlı
        noise_scale = 1.0 + 3.0 * (1.0 - confidence)
        z_lateral = lateral_error_true + self.rng.normal(0.0, self.lateral_noise_std * noise_scale)
        z_heading = heading_error_true + self.rng.normal(0.0, self.heading_noise_std * noise_scale)
        
        return z_lateral, z_heading, confidence, nx, ny, closest_pt[0], closest_pt[1]
    
    def get_visualization_data(self, state):
        """
        Animasyon için kamera görüş alanı verisi.
        Ön kamera konisi (60° açı, 30m menzil).
        """
        x, y, theta, _ = state
        
        # Koni köşe noktaları (otobüs koordinatları)
        half_fov = np.deg2rad(self.fov_deg / 2)
        cone_pts = [
            (x, y),
            (x + self.range_m * np.cos(theta - half_fov),
             y + self.range_m * np.sin(theta - half_fov)),
            (x + self.range_m * np.cos(theta + half_fov),
             y + self.range_m * np.sin(theta + half_fov)),
        ]
        return cone_pts
