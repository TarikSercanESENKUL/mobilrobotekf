import numpy as np

class ExtendedKalmanFilter:
    """
    Otonom otobüs için 6 durumlu Klasik Genişletilmiş Kalman Filtresi (EKF).
    Durum Vektörü x: [X, Y, theta, v, bg, ba]^T
    X, Y: 2B konum koordinatları (m)
    theta: Yönelim açısı (rad)
    v: Boylamsal hız (m/s)
    bg: Jiroskop sapması (rad/s)
    ba: İvmeölçer sapması (m/s^2)
    """
    def __init__(self, wheelbase=6.0, dt=0.1):
        self.L = wheelbase
        self.dt = dt
        
        # Durum vektörü (6x1)
        self.x = np.zeros((6, 1))
        
        # Kovaryans matrisi P (6x6)
        self.P = np.eye(6) * 1.0
        
        # Süreç gürültüsü kovaryansı Q (6x6)
        # Boylamsal hız ve açısal hız gürültüsü IMU standart sapmalarına göre tanımlanır.
        gyro_noise_std = np.deg2rad(0.1)
        accel_noise_std = 0.05
        q_bg = 1e-6  # jiroskop bias random walk gürültüsü
        q_ba = 1e-5  # ivmeölçer bias random walk gürültüsü
        
        self.Q = np.diag([
            0.01,                   # X
            0.01,                   # Y
            (gyro_noise_std**2) * (self.dt**2), # theta
            (accel_noise_std**2) * (self.dt**2), # v
            q_bg,                   # bg
            q_ba                    # ba
        ])
        
        # Sabit Ölçüm Gürültüsü Kovaryansları R (Klasik EKF için)
        self.R_gps = np.diag([0.4**2, 0.4**2])                # GNSS X, Y nominal gürültüsü
        self.R_odom = np.array([[0.08**2]])                   # Odometri hız nominal gürültüsü
        self.R_lidar = np.diag([0.1**2, np.deg2rad(1.0)**2])  # LiDAR range, bearing gürültüsü
        
        # NIS Değerleri (Doğrulama ve Gating için)
        self.last_nis_gps = 0.0
        self.last_nis_odom = 0.0
        self.last_nis_lidar = []

    def init_state(self, x0, y0, theta0, v0, bg0=0.0, ba0=0.0, p_init=1.0):
        """
        Filtre durumunu ve kovaryans matrisini ilklendirir.
        """
        self.x = np.array([[x0], [y0], [theta0], [v0], [bg0], [ba0]], dtype=float)
        self.P = np.eye(6) * p_init
        # Açılar ve biaslar için daha gerçekçi başlangıç belirsizlikleri
        self.P[2, 2] = np.deg2rad(1.0)**2
        self.P[4, 4] = np.deg2rad(0.5)**2
        self.P[5, 5] = 0.1**2

    def predict(self, a_imu, omega_imu):
        """
        EKF Tahmin (Prediction) adımı.
        a_imu: IMU boylamsal ivme ölçümü (m/s^2)
        omega_imu: IMU açısal hız ölçümü (rad/s)
        """
        # Mevcut durum değişkenleri
        theta = self.x[2, 0]
        v = self.x[3, 0]
        bg = self.x[4, 0]
        ba = self.x[5, 0]
        
        # Biaslardan arındırılmış IMU girdileri
        omega_corrected = omega_imu - bg
        a_corrected = a_imu - ba
        
        # 1. Durum Tahmini: x_pred = f(x, u)
        x_pred = self.x[0, 0] + v * np.cos(theta) * self.dt
        y_pred = self.x[1, 0] + v * np.sin(theta) * self.dt
        theta_pred = (theta + omega_corrected * self.dt) % (2.0 * np.pi)
        v_pred = max(0.0, v + a_corrected * self.dt)
        bg_pred = bg
        ba_pred = ba
        
        self.x = np.array([[x_pred], [y_pred], [theta_pred], [v_pred], [bg_pred], [ba_pred]])
        
        # 2. Jacobian Matrisi F = df/dx (6x6)
        F = np.array([
            [1.0, 0.0, -v * np.sin(theta) * self.dt,  np.cos(theta) * self.dt,  0.0,      0.0],
            [0.0, 1.0,  v * np.cos(theta) * self.dt,  np.sin(theta) * self.dt,  0.0,      0.0],
            [0.0, 0.0,  1.0,                          0.0,                     -self.dt,  0.0],
            [0.0, 0.0,  0.0,                          1.0,                      0.0,     -self.dt],
            [0.0, 0.0,  0.0,                          0.0,                      1.0,      0.0],
            [0.0, 0.0,  0.0,                          0.0,                      0.0,      1.0]
        ])
        
        # 3. Kovaryans Tahmini: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Kovaryansın simetrik kalmasını sağlama
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.x.flatten(), self.P

    def update_gps(self, z_gps, R_custom=None):
        """
        GNSS konum güncellemesi (X, Y).
        """
        z = np.array(z_gps, dtype=float).reshape((2, 1))
        
        # Gözlem Matrisi H (2x6)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        R_used = self.R_gps if R_custom is None else np.asarray(R_custom, dtype=float)
        
        # İnovasyon (Innovation) y
        y = z - H @ self.x
        
        # İnovasyon Kovaryansı S
        S = H @ self.P @ H.T + R_used
        
        # Kalman Kazancı K
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
            
        # NIS hesabı (2 Serbestlik Dereceli, %95 Chi-Sq Eşiği = 5.99)
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except np.linalg.LinAlgError:
            nis = np.nan
            
        self.last_nis_gps = nis
        
        # Klasik EKF'te NIS kapısı (Gating) uygulanabilir, ancak aşırı bozulmada güncellemeyi reddeder.
        # Bu projede karşılaştırma için normal gating uygulayacağız.
        if not np.isnan(nis) and nis < 5.99:
            self.x = self.x + K @ y
            self.x[2, 0] %= 2.0 * np.pi  # theta yönelimi sar (wrap)
            
            # Joseph Biçimi Kovaryans Güncellemesi (Kararlılık için)
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_used @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_odom(self, z_v, R_custom=None):
        """
        Odometri hız güncellemesi (v).
        """
        z = np.array([[z_v]], dtype=float)
        
        # Gözlem Matrisi H (1x6)
        H = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        
        R_used = self.R_odom if R_custom is None else np.array([[R_custom]], dtype=float)
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R_used
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
            
        # NIS hesabı (1 Serbestlik Dereceli, %95 Chi-Sq Eşiği = 3.84)
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except np.linalg.LinAlgError:
            nis = np.nan
            
        self.last_nis_odom = nis
        
        if not np.isnan(nis) and nis < 3.84:
            self.x = self.x + K @ y
            self.x[2, 0] %= 2.0 * np.pi
            
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_used @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_lane_camera(self, z_lane, nx, ny, ref_x, ref_y):
        """
        Yol çizgisi izleme kamerasından gelen yanal sapma güncellemesi.
        """
        z = np.array([[z_lane]], dtype=float)
        H = np.array([[nx, ny, 0.0, 0.0, 0.0, 0.0]])
        R_used = np.array([[0.02**2]])
        
        # h(x) tahmini: araç konumu ile referans noktası arasındaki farkın normal vektör üzerine izdüşümü
        h_x = (self.x[0, 0] - ref_x) * nx + (self.x[1, 0] - ref_y) * ny
        y = z - h_x
        
        S = H @ self.P @ H.T + R_used
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
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_used @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_lidar(self, measurements, visible_indices, landmarks, R_custom=None):
        """
        LiDAR / Landmark güncellemesi.
        Birden fazla görünür landmark için ardışık (sequential) güncelleme yapar.
        measurements: list of (dist, bearing) tuples
        visible_indices: list of landmark indices in landmarks array
        landmarks: numpy array of all landmarks (X, Y)
        """
        self.last_nis_lidar = []
        R_used = self.R_lidar if R_custom is None else np.asarray(R_custom, dtype=float)
        
        for (dist, bearing), idx in zip(measurements, visible_indices):
            # Otobüs ve landmark konumları
            x_est = self.x[0, 0]
            y_est = self.x[1, 0]
            theta_est = self.x[2, 0]
            
            lm_x, lm_y = landmarks[idx]
            
            dx = lm_x - x_est
            dy = lm_y - y_est
            d_est = np.hypot(dx, dy)
            
            if d_est < 0.1:
                continue  # Çok yakın durumunda tanımsızlığı önle
                
            # Ölçüm modeli h(x)
            bearing_est = np.arctan2(dy, dx) - theta_est
            bearing_est = (bearing_est + np.pi) % (2.0 * np.pi) - np.pi
            
            h = np.array([[d_est], [bearing_est]])
            
            # Gözlem Matrisi H_i (2x6)
            H = np.array([
                [-dx/d_est,  -dy/d_est,  0.0,  0.0,  0.0,  0.0],
                [dy/(d_est**2), -dx/(d_est**2), -1.0, 0.0,  0.0,  0.0]
            ])
            
            # İnovasyon hesabı
            z = np.array([[dist], [bearing]])
            y = z - h
            y[1, 0] = (y[1, 0] + np.pi) % (2.0 * np.pi) - np.pi  # Açı inovasyonunu sar
            
            S = H @ self.P @ H.T + R_used
            
            try:
                K = self.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = self.P @ H.T @ np.linalg.pinv(S)
                
            # NIS hesabı (2 Serbestlik Dereceli, %95 Chi-Sq Eşiği = 5.99)
            try:
                nis = float((y.T @ np.linalg.inv(S) @ y).item())
            except np.linalg.LinAlgError:
                nis = np.nan
                
            self.last_nis_lidar.append(nis)
            
            # Landmark eşleştirme gating
            if not np.isnan(nis) and nis < 5.99:
                self.x = self.x + K @ y
                self.x[2, 0] %= 2.0 * np.pi
                
                I = np.eye(6)
                self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_used @ K.T
                self.P = 0.5 * (self.P + self.P.T)
                
        return True
