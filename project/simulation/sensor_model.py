import numpy as np

class SensorSimulator:
    """
    Otobüs durumuna göre gürültülü ve bozunmaya uğramış sensör ölçümleri üreten sınıf.
    GNSS bozulma ve tünel kaybı bölgeleri, tekerlek kayması ve rejeneratif frenleme etkileri simüle edilir.
    """
    def __init__(self, landmarks, dt=0.1, rng_seed=42):
        self.dt = dt
        self.landmarks = np.array(landmarks)  # Shape: (M, 2) - Landmark koordinatları [X, Y]
        self.rng = np.random.default_rng(rng_seed)
        
        # Gerçek IMU sapmaları (bias)
        self.true_gyro_bias = 0.03  # rad/s (~1.7 deg/s)
        self.true_accel_bias = 0.15 # m/s^2
        
        # Sensör nominal gürültü parametreleri
        self.gnss_std_normal = 0.4      # m
        self.gnss_std_degraded = 12.0   # m (kentsel kanyon)
        
        self.imu_gyro_noise_nominal = np.deg2rad(0.1)  # rad/s
        self.imu_accel_noise_nominal = 0.05            # m/s^2
        
        self.odom_std_normal = 0.08      # m/s
        self.odom_std_slip = 1.2        # m/s (tekerlek kayması durumu)
        
        self.lidar_range_std_normal = 0.1    # m
        self.lidar_angle_std_normal = np.deg2rad(1.0)  # rad
        
        self.lidar_max_range = 50.0  # LiDAR menzili (m)
        self.lidar_fov = np.pi       # 180 derece görüş açısı (otobüsün ön tarafı)

    def tick_biases(self):
        """
        IMU bias değerlerini yavaşça sürükler (random walk).
        """
        self.true_gyro_bias += self.rng.normal(0.0, 0.0001)
        self.true_accel_bias += self.rng.normal(0.0, 0.0005)

    def get_gnss(self, state, s, tunnel_mode=False):
        """
        GNSS konum ölçümlerini üretir.
        tunnel_mode=True ve s ∈ [300, 500] ise sinyal tamamen kaybolur (None döner).
        degraded ise gürültü standart sapması artar.
        """
        x, y, _, _ = state
        s_mod = s % 1114.0
        
        # Bozulma veya kesinti bölgesi tespiti
        in_zone = (300.0 <= s_mod <= 500.0)
        
        if in_zone and tunnel_mode:
            # Tünel senaryosu: Sinyal tamamen kesildi
            return None, True, np.inf
            
        std = self.gnss_std_degraded if (in_zone and not tunnel_mode) else self.gnss_std_normal
        
        # Ölçüm üretimi
        z_x = x + self.rng.normal(0.0, std)
        z_y = y + self.rng.normal(0.0, std)
        
        return np.array([z_x, z_y]), in_zone, std

    def get_imu(self, true_steering, true_accel, true_v, is_regenerating=False):
        """
        IMU ivmeölçer ve jiroskop ölçümlerini üretir.
        is_regenerating=True ise frenleme torkları nedeniyle IMU gürültü varyansı artar.
        """
        # Jiroskop ölçümü: omega_true = (v/L)*tan(phi)
        L = 6.0
        phi_clipped = np.clip(true_steering, -np.deg2rad(35.0), np.deg2rad(35.0))
        omega_true = (true_v / L) * np.tan(phi_clipped)
        
        # Rejeneratif frenleme anlarında IMU gürültü artışı
        noise_scale = 1.5 if is_regenerating else 1.0
        gyro_noise = self.imu_gyro_noise_nominal * noise_scale
        accel_noise = self.imu_accel_noise_nominal * noise_scale
        
        # Gürültülü ve bias eklenmiş ölçümler
        omega_imu = omega_true + self.true_gyro_bias + self.rng.normal(0.0, gyro_noise)
        a_imu = true_accel + self.true_accel_bias + self.rng.normal(0.0, accel_noise)
        
        return np.array([a_imu, omega_imu])

    def get_odometry(self, state, s, is_regenerating=False):
        """
        Odometri hız ölçümlerini üretir.
        s ∈ [700, 800] bölgesinde tekerlek kayması (slip) simüle edilir.
        is_regenerating=True ise rejeneratif frenleme nedeniyle tekerlek odometrisi gürültü varyansı artar.
        """
        _, _, _, v = state
        s_mod = s % 1114.0
        is_slipping = (700.0 <= s_mod <= 800.0)
        
        if is_slipping:
            # Kayma bölgesi
            slip_bias = -2.5
            std = self.odom_std_slip
            z_v = max(0.0, v + slip_bias + self.rng.normal(0.0, std))
        else:
            # Rejeneratif frenleme anlarında tekerlek kayması/gürültü artışı
            noise_scale = 1.8 if is_regenerating else 1.0
            std = self.odom_std_normal * noise_scale
            z_v = max(0.0, v + self.rng.normal(0.0, std))
            
        return z_v, is_slipping, std

    def get_lidar(self, state):
        """
        Otobüsün etrafındaki görünür landmarklar için uzaklık ve kerteriz açısı ölçümlerini döndürür.
        Algılanan landmark sayısı azaldığında LiDAR ölçüm gürültüsü artırılır.
        """
        x, y, theta, _ = state
        
        visible_indices = []
        measurements = []
        
        for i, lm in enumerate(self.landmarks):
            dx = lm[0] - x
            dy = lm[1] - y
            dist = np.hypot(dx, dy)
            
            if dist <= self.lidar_max_range:
                global_angle = np.arctan2(dy, dx)
                bearing = (global_angle - theta + np.pi) % (2.0 * np.pi) - np.pi
                
                # FOV kontrolü (ön taraftaki 180 derece)
                if abs(bearing) <= self.lidar_fov / 2.0:
                    visible_indices.append(i)
                    measurements.append((dist, bearing))
                    
        num_visible = len(measurements)
        
        if num_visible == 0:
            return [], [], 0
            
        # Landmark sayısına bağlı gürültü ölçekleme
        if num_visible <= 2:
            noise_scale = 3.0
        else:
            noise_scale = 1.0
            
        r_std = self.lidar_range_std_normal * noise_scale
        b_std = self.lidar_angle_std_normal * noise_scale
        
        noisy_measurements = []
        for dist, bearing in measurements:
            z_d = dist + self.rng.normal(0.0, r_std)
            z_b = (bearing + self.rng.normal(0.0, b_std) + np.pi) % (2.0 * np.pi) - np.pi
            noisy_measurements.append((z_d, z_b))
            
        return noisy_measurements, visible_indices, num_visible

    def get_lane_tracking(self, state, route_waypoints):
        """
        Yol çizgisi izleme kamerasından gelen yanal sapma ölçümü.
        Fiziksel olarak otobüsün şerit merkezinden olan lateral sapmasını ölçer.
        """
        x, y, theta, _ = state
        # En yakın waypoint'i bulalım
        dists = np.hypot(route_waypoints[:, 0] - x, route_waypoints[:, 1] - y)
        idx = np.argmin(dists)
        closest_pt = route_waypoints[idx]
        
        # Lateral hata hesabı: Teğet-Normal izdüşümü ile
        if idx < len(route_waypoints) - 1:
            tx = route_waypoints[idx+1, 0] - closest_pt[0]
            ty = route_waypoints[idx+1, 1] - closest_pt[1]
        else:
            tx = closest_pt[0] - route_waypoints[idx-1, 0]
            ty = closest_pt[1] - route_waypoints[idx-1, 1]
            
        t_len = np.hypot(tx, ty)
        if t_len > 1e-6:
            tx, ty = tx/t_len, ty/t_len
        else:
            tx, ty = 1.0, 0.0
            
        # Normal vektör (sol taraf): (-ty, tx)
        nx = -ty
        ny = tx
        
        # Araç pozisyonu ile en yakın nokta arasındaki vektör
        dx = x - closest_pt[0]
        dy = y - closest_pt[1]
        
        # Yanal hata (lateral displacement): normal vektör üzerine izdüşüm
        lateral_error_true = dx * nx + dy * ny
        
        # Çok hassas şerit takip gürültüsü (2 cm std dev)
        z_lane = lateral_error_true + self.rng.normal(0.0, 0.02)
        
        return z_lane, nx, ny, closest_pt[0], closest_pt[1]

    def get_proximity(self, state, active_stop_position):
        """
        Durak platformuna olan mesafeyi ölçen ultrasonik yakınlık sensörü.
        Yalnızca durağa 20 m mesafe içindeyken veri üretir.
        """
        x, y, _, _ = state
        dist = np.hypot(active_stop_position[0] - x, active_stop_position[1] - y)
        if dist <= 20.0:
            z_prox = dist + self.rng.normal(0.0, 0.03) # 3 cm gürültülü
            return z_prox
        return None

