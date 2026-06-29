"""
Lane Model
==========
Şerit merkezi metrikleri ve şeritte kalma performans hesaplaması.

Çift şeritli yol yapısı:
- Sağ şerit (otobüs şeridi): rota çizgisi üzerinde
- Sol şerit (karşı yön): rota çizgisinden -3.6m uzakta
- Şerit genişliği: 3.6 m
- Sağ şerit iç sınırı: +1.8m
- Orta şerit çizgisi: -1.8m  
- Sol şerit dış sınırı: -5.4m
"""
import numpy as np


class LaneModel:
    """
    Şerit yapısı ve lateral hata metrikleri.
    
    Otobüs rota çizgisinin TAM üzerinde ilerliyor.
    Şerit ihlali: |lateral_error| > lane_width/2 = 1.8m
    """
    
    lane_width = 3.6    # m — standart şehir içi otobüs şeridi
    road_width = 7.2    # m — toplam yol genişliği (2 şerit)
    
    def __init__(self, route_waypoints):
        self.route_waypoints = route_waypoints
        
        # Metrik takip dizileri
        self.lateral_errors = []
        self.lane_departures = []     # Şerit dışı olan adımlar
        self.timestamps = []
        
    def compute_lateral_error(self, state):
        """
        Araç ile şerit merkezi arasındaki yanal hata.
        
        Sağ şerit merkezi = rota çizgisi (route_waypoints)
        
        Returns:
            lateral_error (float): Pozitif = sağ taraf dışı, Negatif = sol taraf dışı
            nx, ny (float): Şerit normal vektörü
        """
        x, y, theta, _ = state
        
        dists = np.hypot(self.route_waypoints[:, 0] - x, 
                          self.route_waypoints[:, 1] - y)
        idx = np.argmin(dists)
        closest_pt = self.route_waypoints[idx]
        
        # Teğet vektörü
        if idx < len(self.route_waypoints) - 1:
            tx = self.route_waypoints[idx+1, 0] - closest_pt[0]
            ty = self.route_waypoints[idx+1, 1] - closest_pt[1]
        else:
            tx = closest_pt[0] - self.route_waypoints[idx-1, 0]
            ty = closest_pt[1] - self.route_waypoints[idx-1, 1]
            
        t_len = np.hypot(tx, ty)
        if t_len > 1e-6:
            tx, ty = tx / t_len, ty / t_len
            
        nx = -ty
        ny = tx
        
        dx = x - closest_pt[0]
        dy = y - closest_pt[1]
        lateral_error = dx * nx + dy * ny
        
        return lateral_error, nx, ny
    
    def is_lane_departure(self, lateral_error):
        """Şerit ihlali var mı?"""
        return abs(lateral_error) > (self.lane_width / 2.0)
    
    def record_step(self, state, timestamp):
        """Her simülasyon adımında lateral hata kaydet."""
        lateral_error, _, _ = self.compute_lateral_error(state)
        self.lateral_errors.append(lateral_error)
        self.lane_departures.append(self.is_lane_departure(lateral_error))
        self.timestamps.append(timestamp)
        
    def get_metrics(self):
        """Şerit takip metrikleri hesapla."""
        if not self.lateral_errors:
            return {}
            
        errors = np.abs(self.lateral_errors)
        departures = np.array(self.lane_departures)
        n_total = len(errors)
        n_departures = int(np.sum(departures))
        
        return {
            'mean_lane_center_error_m': float(np.mean(errors)),
            'max_lane_center_error_m': float(np.max(errors)),
            'lane_departure_count': n_departures,
            'lane_keeping_success_rate_percent': float((n_total - n_departures) / n_total * 100),
        }
    
    @staticmethod
    def get_lane_boundaries_at_point(route_point, nx, ny):
        """
        Belirli bir noktada şerit sınırlarını döndür.
        
        Returns:
            right_edge: Sağ kenar koordinatı
            center_line: Orta şerit çizgisi koordinatı
            left_edge: Sol kenar koordinatı
            left_outer: Sol karşı şerit dış kenar koordinatı
        """
        rx, ry = route_point
        hw = 1.8  # half lane width
        
        right_edge = (rx + hw * nx, ry + hw * ny)
        center_line = (rx - hw * nx, ry - hw * ny)
        left_edge = (rx - 3 * hw * nx, ry - 3 * hw * ny)
        left_outer = (rx - 3 * hw * nx, ry - 3 * hw * ny)
        
        return right_edge, center_line, left_edge
