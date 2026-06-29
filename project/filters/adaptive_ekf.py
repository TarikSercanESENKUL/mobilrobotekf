import numpy as np
from project.filters.ekf import ExtendedKalmanFilter

class AdaptiveExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Ölçüm gürültüsü kovaryansı R_k ve süreç gürültüsü Q_k matrislerini
    sensör güvenilirliği, sürüş koşulları ve ortam durumuna göre dinamik
    olarak güncelleyen Adaptif EKF (AEKF) sınıfı.
    """
    def __init__(self, wheelbase=6.0, dt=0.1, route_waypoints=None):
        super().__init__(wheelbase=wheelbase, dt=dt)
        self.route_waypoints = route_waypoints  # Shape: (W, 2) - [X, Y] koordinatları
        
        # LiDAR nominal gürültü standart sapmaları
        self.lidar_range_std_normal = 0.1
        self.lidar_angle_std_normal = np.deg2rad(1.0)
        
        # Süreç gürültüsünü uyarlamak için takip değişkenleri
        self.degraded_time = 0.0
        self.nominal_Q = self.Q.copy()
        
        # Kaydedilecek dinamik R değerleri izleri (grafik çizimi için)
        self.r_gps_trace = []
        self.r_odom_trace = []
        self.r_lidar_trace = []
        self.q_theta_trace = []

    def get_closest_route_distance(self):
        """
        Kestirilen otobüs konumunun bilinen rota koridoruna en yakın mesafesini hesaplar.
        """
        if self.route_waypoints is None:
            return 0.0
        
        x_est = self.x[0, 0]
        y_est = self.x[1, 0]
        
        # En yakın noktaya olan mesafeyi bul
        dists = np.hypot(self.route_waypoints[:, 0] - x_est, self.route_waypoints[:, 1] - y_est)
        min_dist = np.min(dists)
        
        return min_dist

    def check_route_corridor(self, threshold=4.0):
        """
        Otobüs rota koridorundan uzaklaşırsa hata/uyarı üretir.
        """
        dist = self.get_closest_route_distance()
        if dist > threshold:
            # Akademik çıktı için print uyarısı üretelim
            print(f"[KORİDOR UYARISI] Araç rotadan {dist:.2f} m saptı! Güvenlik toleransı ({threshold} m) aşıldı.")
            return True, dist
        return False, dist

    def adapt_process_noise(self, gnss_degraded):
        """
        IMU drift etkisinin birikmesini önlemek için GNSS bozulma süresi uzadıkça
        süreç gürültüsü kovaryansını (Q) dinamik olarak artırır.
        """
        if gnss_degraded:
            self.degraded_time += self.dt
        else:
            self.degraded_time = 0.0
            
        # GNSS bozuldukça bias ve theta süreç gürültüsünü artır (belirsizlik artsın ki GNSS gelince hızlı düzelsin)
        scale = 1.0 + 0.1 * self.degraded_time
        
        self.Q = self.nominal_Q.copy()
        self.Q[2, 2] = self.nominal_Q[2, 2] * scale  # theta süreç gürültüsü
        self.Q[4, 4] = self.nominal_Q[4, 4] * scale  # gyro bias gürültüsü
        self.Q[5, 5] = self.nominal_Q[5, 5] * scale  # accel bias gürültüsü
        
        self.q_theta_trace.append(self.Q[2, 2])

    def update_gps_adaptive(self, z_gps, is_degraded):
        """
        GNSS konum güncellemesi.
        Kovaryans hem bozulma bölgesine hem de inovasyon residual değerine göre uyarlanır.
        """
        z = np.array(z_gps, dtype=float).reshape((2, 1))
        H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        # İnovasyon (residual) hesabı
        y = z - H @ self.x
        y_norm = np.linalg.norm(y)
        
        # Temel gürültü varyansı belirleme
        if is_degraded:
            R_gps_base = 12.0**2  # 144.0 m^2
        else:
            R_gps_base = 0.4**2   # 0.16 m^2
            
        # Residual normuna bağlı olarak kovaryansı ölçeklendir (Sage-Husa benzeri inovasyon uyarlaması)
        R_gps_val = R_gps_base * (1.0 + 0.5 * (y_norm**2))
        
        # Fiziksel üst ve alt sınırlar koyalım
        R_gps_val = np.clip(R_gps_val, 0.04, 800.0)
        
        R_adapted = np.diag([R_gps_val, R_gps_val])
        self.r_gps_trace.append(R_gps_val)
        
        S = H @ self.P @ H.T + R_adapted
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
            
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except np.linalg.LinAlgError:
            nis = np.nan
            
        self.last_nis_gps = nis
        
        # Bozulma bölgesindeysek kapıyı esnetelim
        gate_threshold = 30.0 if is_degraded else 5.99
        
        if not np.isnan(nis) and nis < gate_threshold:
            self.x = self.x + K @ y
            self.x[2, 0] %= 2.0 * np.pi
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_adapted @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_odom_adaptive(self, z_v, is_slipping, is_near_stop):
        """
        Odometri güncellemesi.
        Kovaryans tekerlek kayması ve odometri ile predicted hız arasındaki uyumsuzluğa göre uyarlanır.
        """
        v_est = self.x[3, 0]
        
        # Ölçüm ile tahmin arasındaki hız farkı
        e_v = abs(z_v - v_est)
        
        # Temel varyans belirleme
        if is_slipping:
            R_odom_base = 1.2**2  # 1.44
        elif is_near_stop and v_est < 1.5:
            # Durakta dururken/yavaşlarken tekerlek kayması çok azdır -> yüksek güven
            R_odom_base = 0.02**2 # 0.0004
        else:
            R_odom_base = 0.08**2 # 0.0064
            
        # Hız farkına göre kovaryansı ölçeklendir
        R_odom_val = R_odom_base * (1.0 + 3.0 * (e_v**2))
        R_odom_val = np.clip(R_odom_val, 0.0001, 10.0)
        
        R_adapted = np.array([[R_odom_val]])
        self.r_odom_trace.append(R_odom_val)
        
        z = np.array([[z_v]], dtype=float)
        H = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R_adapted
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
            
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except np.linalg.LinAlgError:
            nis = np.nan
            
        self.last_nis_odom = nis
        
        gate_threshold = 15.0 if (is_slipping or is_near_stop) else 3.84
        
        if not np.isnan(nis) and nis < gate_threshold:
            self.x = self.x + K @ y
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_adapted @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_lane_camera_adaptive(self, z_lane, nx, ny, ref_x, ref_y):
        """
        Yol çizgisi izleme kamerasından gelen yanal sapma güncellemesi (Adaptif).
        """
        z = np.array([[z_lane]], dtype=float)
        H = np.array([[nx, ny, 0.0, 0.0, 0.0, 0.0]])
        
        # h(x) tahmini
        h_x = (self.x[0, 0] - ref_x) * nx + (self.x[1, 0] - ref_y) * ny
        y = z - h_x
        
        R_base = 0.02**2
        R_adapted_val = R_base * (1.0 + 5.0 * (y[0, 0]**2))
        R_adapted = np.array([[R_adapted_val]])
        
        S = H @ self.P @ H.T + R_adapted
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
            
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except np.linalg.LinAlgError:
            nis = np.nan
            
        if not np.isnan(nis) and nis < 3.84:
            self.x = self.x + K @ y
            self.x[2, 0] %= 2.0 * np.pi
            
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_adapted @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_lidar_adaptive(self, measurements, visible_indices, landmarks, num_visible, is_near_stop):
        """
        LiDAR güncellemesi.
        Kovaryans, algılanan landmark sayısına göre değişir.
        """
        v_est = self.x[3, 0]
        self.last_nis_lidar = []
        
        if num_visible <= 0:
            self.r_lidar_trace.append(self.R_lidar[0, 0])
            return False
            
        # Temel standard sapmaları belirle
        if is_near_stop and v_est < 1.5:
            # Durakta yanaşırken landmark ölçümlerine çok güven
            range_std_base = self.lidar_range_std_normal * 0.25  # 0.025 m
            angle_std_base = self.lidar_angle_std_normal * 0.25  # 0.25 deg
        else:
            range_std_base = self.lidar_range_std_normal
            angle_std_base = self.lidar_angle_std_normal
            
        # Görünür landmark sayısına göre kovaryansı ölçekle
        # Landmark sayısı azaldıkça kovaryans artar, arttıkça nominale yaklaşır
        scale_factor = 4.0 / max(1.0, float(num_visible))
        
        R_adapted = np.diag([
            (range_std_base**2) * scale_factor,
            (angle_std_base**2) * scale_factor
        ])
        
        self.r_lidar_trace.append((range_std_base**2) * scale_factor)
        
        # Sıralı güncelleme
        for (dist, bearing), idx in zip(measurements, visible_indices):
            x_est = self.x[0, 0]
            y_est = self.x[1, 0]
            theta_est = self.x[2, 0]
            
            lm_x, lm_y = landmarks[idx]
            dx = lm_x - x_est
            dy = lm_y - y_est
            d_est = np.hypot(dx, dy)
            
            if d_est < 0.1:
                continue
                
            bearing_est = np.arctan2(dy, dx) - theta_est
            bearing_est = (bearing_est + np.pi) % (2.0 * np.pi) - np.pi
            
            h = np.array([[d_est], [bearing_est]])
            
            H = np.array([
                [-dx/d_est,  -dy/d_est,  0.0,  0.0,  0.0,  0.0],
                [dy/(d_est**2), -dx/(d_est**2), -1.0, 0.0,  0.0,  0.0]
            ])
            
            z = np.array([[dist], [bearing]])
            y = z - h
            y[1, 0] = (y[1, 0] + np.pi) % (2.0 * np.pi) - np.pi
            
            S = H @ self.P @ H.T + R_adapted
            
            try:
                K = self.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = self.P @ H.T @ np.linalg.pinv(S)
                
            try:
                nis = float((y.T @ np.linalg.inv(S) @ y).item())
            except np.linalg.LinAlgError:
                nis = np.nan
                
            self.last_nis_lidar.append(nis)
            
            # Gating eşiği (adaptif durumda biraz daha esnek)
            gate_threshold = 12.0 if num_visible <= 2 else 5.99
            
            if not np.isnan(nis) and nis < gate_threshold:
                self.x = self.x + K @ y
                self.x[2, 0] %= 2.0 * np.pi
                I = np.eye(6)
                self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_adapted @ K.T
                self.P = 0.5 * (self.P + self.P.T)
                
        return True
