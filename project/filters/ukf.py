import numpy as np

class UnscentedKalmanFilter:
    """
    Otonom otobüs için 6 durumlu Unscented Kalman Filtresi (UKF).
    Durum Vektörü x: [X, Y, theta, v, bg, ba]^T
    Doğrusal olmayan dinamikleri Jacobian hesaplamadan sigma noktalarıyla modeller.
    """
    def __init__(self, wheelbase=6.0, dt=0.1, enable_gating=True):
        self.L = wheelbase
        self.dt = dt
        self.enable_gating = enable_gating
        
        # Durum Boyutu
        self.n = 6
        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n) * 1.0
        
        # Süreç Gürültüsü Q (6x6)
        gyro_noise_std = np.deg2rad(0.1)
        accel_noise_std = 0.05
        q_bg = 1e-6
        q_ba = 1e-5
        self.Q = np.diag([
            0.01,
            0.01,
            (gyro_noise_std**2) * (self.dt**2),
            (accel_noise_std**2) * (self.dt**2),
            q_bg,
            q_ba
        ])
        
        # Ölçüm gürültüleri
        self.R_gps = np.diag([0.4**2, 0.4**2])
        self.R_odom = np.array([[0.08**2]])
        self.R_lidar = np.diag([0.1**2, np.deg2rad(1.0)**2])
        
        # UKF Parametreleri (Standard Van der Merwe)
        self.alpha = 0.001
        self.beta = 2.0
        self.kappa = 0.0
        
        # Ağırlıkların hesaplanması
        self.lmbda = (self.alpha**2) * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lmbda)
        
        # Ağırlık dizileri
        self.w_m = np.zeros(2 * self.n + 1)
        self.w_c = np.zeros(2 * self.n + 1)
        
        self.w_m[0] = self.lmbda / (self.n + self.lmbda)
        self.w_c[0] = self.w_m[0] + (1.0 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.w_m[i] = 0.5 / (self.n + self.lmbda)
            self.w_c[i] = self.w_m[i]

    def init_state(self, x0, y0, theta0, v0, bg0=0.0, ba0=0.0, p_init=1.0):
        """
        Filtre durumunu ve kovaryans matrisini ilklendirir.
        """
        self.x = np.array([[x0], [y0], [theta0], [v0], [bg0], [ba0]], dtype=float)
        self.P = np.eye(self.n) * p_init
        self.P[2, 2] = np.deg2rad(1.0)**2
        self.P[4, 4] = np.deg2rad(0.5)**2
        self.P[5, 5] = 0.1**2

    def _get_sqrt_P(self):
        """
        Kovaryans matrisinin karekökünü (SVD veya Cholesky ile) kararlı bir şekilde hesaplar.
        """
        try:
            # Cholesky ayrışımı dene
            return np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            # SVD ile karekök al (sayısal kararlılık için)
            U, s, Vt = np.linalg.svd(self.P)
            s[s < 0] = 0.0
            return U @ np.diag(np.sqrt(s))

    def _generate_sigma_points(self):
        """
        Mevcut durum ve kovaryansa göre 2n+1 adet sigma noktası üretir.
        """
        sigma_points = np.zeros((self.n, 2 * self.n + 1))
        sigma_points[:, 0:1] = self.x
        
        sqrt_P = self._get_sqrt_P()
        
        for i in range(self.n):
            col = sqrt_P[:, i:i+1]
            sigma_points[:, i + 1 : i + 2] = self.x + self.gamma * col
            sigma_points[:, i + 1 + self.n : i + 2 + self.n] = self.x - self.gamma * col
            
        return sigma_points

    def predict(self, a_imu, omega_imu):
        """
        UKF Tahmin adımı. Sigma noktalarını araç modeli boyunca ilerletir.
        """
        # 1. Sigma noktalarını üret
        sigmas = self._generate_sigma_points()
        sigmas_pred = np.zeros_like(sigmas)
        
        # 2. Her sigma noktasını model üzerinden ilerlet
        for i in range(2 * self.n + 1):
            theta = sigmas[2, i]
            v = sigmas[3, i]
            bg = sigmas[4, i]
            ba = sigmas[5, i]
            
            omega_corrected = omega_imu - bg
            a_corrected = a_imu - ba
            
            sigmas_pred[0, i] = sigmas[0, i] + v * np.cos(theta) * self.dt
            sigmas_pred[1, i] = sigmas[1, i] + v * np.sin(theta) * self.dt
            sigmas_pred[2, i] = (theta + omega_corrected * self.dt) % (2.0 * np.pi)
            sigmas_pred[3, i] = max(0.0, v + a_corrected * self.dt)
            sigmas_pred[4, i] = bg
            sigmas_pred[5, i] = ba
            
        # 3. Kestirilen ortalama durumu hesapla (Açısal durum için dairesel ortalama)
        x_pred = np.zeros((self.n, 1))
        
        # Konum, Hız ve Biaslar için normal ortalama
        for i in [0, 1, 3, 4, 5]:
            x_pred[i, 0] = np.sum(self.w_m * sigmas_pred[i, :])
            
        # Açı için dairesel ortalama
        sin_sum = np.sum(self.w_m * np.sin(sigmas_pred[2, :]))
        cos_sum = np.sum(self.w_m * np.cos(sigmas_pred[2, :]))
        x_pred[2, 0] = np.arctan2(sin_sum, cos_sum) % (2.0 * np.pi)
        
        # 4. Kestirilen Kovaryansı hesapla (Açı farkını sararak)
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = sigmas_pred[:, i:i+1] - x_pred
            # Açı farkını [-pi, pi] aralığına sar
            diff[2, 0] = (diff[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
            P_pred += self.w_c[i] * (diff @ diff.T)
            
        self.x = x_pred
        self.P = P_pred + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.x.flatten(), self.P

    def update_gps(self, z_gps):
        """
        GNSS güncellemesi.
        """
        z = np.array(z_gps, dtype=float).reshape((2, 1))
        
        # Sigma noktalarını yeniden üret
        sigmas = self._generate_sigma_points()
        
        # Gözlem modeli: h(x) = [x, y]^T
        sigmas_z = sigmas[0:2, :]
        
        # Tahmini ölçüm ortalaması
        z_pred = np.zeros((2, 1))
        z_pred[0, 0] = np.sum(self.w_m * sigmas_z[0, :])
        z_pred[1, 0] = np.sum(self.w_m * sigmas_z[1, :])
        
        # Ölçüm kovaryansı P_zz
        P_zz = np.zeros((2, 2))
        for i in range(2 * self.n + 1):
            diff_z = sigmas_z[:, i:i+1] - z_pred
            P_zz += self.w_c[i] * (diff_z @ diff_z.T)
        P_zz += self.R_gps
        
        # İnovasyon (residual) hesabı
        innovation = z - z_pred
        
        # Mahalanobis kapılama (gating) testi
        if self.enable_gating:
            try:
                d2 = float((innovation.T @ np.linalg.inv(P_zz) @ innovation).item())
            except np.linalg.LinAlgError:
                try:
                    d2 = float((innovation.T @ np.linalg.pinv(P_zz) @ innovation).item())
                except:
                    d2 = 999.0
                    
            if d2 > 9.21:
                # Outlier ölçüm: Güncellemeyi atla ve durum güncellemesi yapma
                return False
            
        # Çapraz kovaryans P_xz
        P_xz = np.zeros((self.n, 2))
        for i in range(2 * self.n + 1):
            diff_x = sigmas[:, i:i+1] - self.x
            diff_x[2, 0] = (diff_x[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
            diff_z = sigmas_z[:, i:i+1] - z_pred
            P_xz += self.w_c[i] * (diff_x @ diff_z.T)
            
        # Kalman Kazancı K
        try:
            K = P_xz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            K = P_xz @ np.linalg.pinv(P_zz)
            
        # Durum ve kovaryans güncelleme
        self.x = self.x + K @ innovation
        self.x[2, 0] %= 2.0 * np.pi
        
        self.P = self.P - K @ P_zz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_odom(self, z_v):
        """
        Odometri hız güncellemesi.
        """
        z = np.array([[z_v]], dtype=float)
        
        sigmas = self._generate_sigma_points()
        
        # Gözlem modeli: h(x) = v
        sigmas_z = sigmas[3:4, :]
        
        z_pred = np.array([[np.sum(self.w_m * sigmas_z[0, :])]])
        
        # P_zz
        P_zz = np.zeros((1, 1))
        for i in range(2 * self.n + 1):
            diff_z = sigmas_z[:, i:i+1] - z_pred
            P_zz += self.w_c[i] * (diff_z @ diff_z.T)
        P_zz += self.R_odom
        
        # P_xz
        P_xz = np.zeros((self.n, 1))
        for i in range(2 * self.n + 1):
            diff_x = sigmas[:, i:i+1] - self.x
            diff_x[2, 0] = (diff_x[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
            diff_z = sigmas_z[:, i:i+1] - z_pred
            P_xz += self.w_c[i] * (diff_x @ diff_z.T)
            
        try:
            K = P_xz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            K = P_xz @ np.linalg.pinv(P_zz)
            
        innovation = z - z_pred
        self.x = self.x + K @ innovation
        self.x[2, 0] %= 2.0 * np.pi
        
        self.P = self.P - K @ P_zz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_lane_camera(self, z_lane, nx, ny, ref_x, ref_y):
        """
        Yol çizgisi izleme kamerasından gelen yanal sapma güncellemesi (UKF).
        """
        z = np.array([[z_lane]], dtype=float)
        sigmas = self._generate_sigma_points()
        
        # Gözlem modeli: h(x) = (x_pos - ref_x) * nx + (y_pos - ref_y) * ny
        sigmas_z = (sigmas[0:1, :] - ref_x) * nx + (sigmas[1:2, :] - ref_y) * ny
        
        z_pred = np.array([[np.sum(self.w_m * sigmas_z[0, :])]])
        
        R_used = np.array([[0.02**2]])
        
        P_zz = np.zeros((1, 1))
        for i in range(2 * self.n + 1):
            diff_z = sigmas_z[:, i:i+1] - z_pred
            P_zz += self.w_c[i] * (diff_z @ diff_z.T)
        P_zz += R_used
        
        P_xz = np.zeros((self.n, 1))
        for i in range(2 * self.n + 1):
            diff_x = sigmas[:, i:i+1] - self.x
            diff_x[2, 0] = (diff_x[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
            diff_z = sigmas_z[:, i:i+1] - z_pred
            P_xz += self.w_c[i] * (diff_x @ diff_z.T)
            
        try:
            K = P_xz @ np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            K = P_xz @ np.linalg.pinv(P_zz)
            
        innovation = z - z_pred
        
        try:
            nis = float((innovation.T @ np.linalg.inv(P_zz) @ innovation).item())
        except np.linalg.LinAlgError:
            nis = np.nan
            
        if not np.isnan(nis) and nis < 3.84:
            self.x = self.x + K @ innovation
            self.x[2, 0] %= 2.0 * np.pi
            self.P = self.P - K @ P_zz @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            return True
        return False

    def update_lidar(self, measurements, visible_indices, landmarks):
        """
        LiDAR landmark güncellemesi. Ardışık olarak güncellenir.
        """
        for (dist, bearing), idx in zip(measurements, visible_indices):
            lm_x, lm_y = landmarks[idx]
            
            sigmas = self._generate_sigma_points()
            sigmas_z = np.zeros((2, 2 * self.n + 1))
            
            # Her sigma noktası için tahmini gözlem değerini hesapla
            for i in range(2 * self.n + 1):
                dx = lm_x - sigmas[0, i]
                dy = lm_y - sigmas[1, i]
                theta = sigmas[2, i]
                
                d_est = np.hypot(dx, dy)
                bearing_est = np.arctan2(dy, dx) - theta
                bearing_est = (bearing_est + np.pi) % (2.0 * np.pi) - np.pi
                
                sigmas_z[0, i] = d_est
                sigmas_z[1, i] = bearing_est
                
            # Ölçüm tahmini ortalaması (Açı için dairesel ortalama)
            z_pred = np.zeros((2, 1))
            z_pred[0, 0] = np.sum(self.w_m * sigmas_z[0, :])
            
            sin_sum = np.sum(self.w_m * np.sin(sigmas_z[1, :]))
            cos_sum = np.sum(self.w_m * np.cos(sigmas_z[1, :]))
            z_pred[1, 0] = np.arctan2(sin_sum, cos_sum)
            
            # P_zz
            P_zz = np.zeros((2, 2))
            for i in range(2 * self.n + 1):
                diff_z = sigmas_z[:, i:i+1] - z_pred
                diff_z[1, 0] = (diff_z[1, 0] + np.pi) % (2.0 * np.pi) - np.pi
                P_zz += self.w_c[i] * (diff_z @ diff_z.T)
            P_zz += self.R_lidar
            
            # P_xz
            P_xz = np.zeros((self.n, 2))
            for i in range(2 * self.n + 1):
                diff_x = sigmas[:, i:i+1] - self.x
                diff_x[2, 0] = (diff_x[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
                diff_z = sigmas_z[:, i:i+1] - z_pred
                diff_z[1, 0] = (diff_z[1, 0] + np.pi) % (2.0 * np.pi) - np.pi
                P_xz += self.w_c[i] * (diff_x @ diff_z.T)
                
            try:
                K = P_xz @ np.linalg.inv(P_zz)
            except np.linalg.LinAlgError:
                K = P_xz @ np.linalg.pinv(P_zz)
                
            z = np.array([[dist], [bearing]])
            innovation = z - z_pred
            innovation[1, 0] = (innovation[1, 0] + np.pi) % (2.0 * np.pi) - np.pi
            
            self.x = self.x + K @ innovation
            self.x[2, 0] %= 2.0 * np.pi
            
            self.P = self.P - K @ P_zz @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            
        return True
