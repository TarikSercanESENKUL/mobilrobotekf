import numpy as np

class StopApproachEnv:
    """
    Otonom otobüsün durak yaklaşma, durma ve durakta 15 saniye bekleme (dwell) 
    davranışını simüle eden 1D pekiştirmeli öğrenme ortamı.
    """
    def __init__(self, dt=0.1, max_steps=500):
        self.dt = dt
        self.max_steps = max_steps
        
        # Sınırlar ve normalleştirme katsayıları
        self.max_distance = 40.0
        self.max_velocity = 10.0
        self.max_accel = 2.5
        
        # Ayrık aksiyon ivmeleri:
        # 0: Güçlü fren (-2.5 m/s^2), 1: Hafif fren (-1.0 m/s^2), 
        # 2: Sabit hız (0.0 m/s^2), 3: Hafif hızlanma (+0.5 m/s^2)
        self.action_space = [-2.5, -1.0, 0.0, 0.5]
        self.action_size = len(self.action_space)
        
        # Durum Boyutu:
        # 0: d_stop_norm (Durağa kalan mesafe)
        # 1: v_norm (Boylamsal Hız)
        # 2: a_prev_norm (Önceki ivme aksiyonu)
        # 3: ekf_position_error_norm (Lokalizasyon hatası)
        # 4: is_stop_zone (Son 15m bölgesinde olma durumu)
        # 5: dwell_ratio (Bekleme süresi tamamlanma oranı)
        self.state_size = 6
        
        self.dwell_time = 15.0         # saniye (Durakta bekleme süresi)
        self.stop_tolerance = 0.5      # metre (Duruş hassasiyeti)
        self.stop_velocity_threshold = 0.3  # m/s (biraz daha geniş eşik)
        
        self.reset()

        
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
            
        # Başlangıç durumu (Durağa kalan mesafe ~35m, hız ~8 m/s)
        self.d_stop = 35.0 + self.rng.uniform(-3.0, 3.0)
        self.v = 8.0 + self.rng.uniform(-1.0, 1.0)
        self.a_prev = 0.0
        self.dwell_counter = 0.0
        
        # Sentetik Adaptif EKF konum kestirim hatası
        self.ekf_error = np.clip(0.12 + self.rng.normal(0.0, 0.05), 0.02, 0.4)
        
        self.step_count = 0
        self.done = False
        
        return self._get_obs()
        
    def _get_obs(self):
        # Durum normalleştirme
        d_norm = self.d_stop / self.max_distance
        v_norm = self.v / self.max_velocity
        a_norm = self.a_prev / self.max_accel
        ekf_err_norm = self.ekf_error / 0.5
        is_stop_zone = 1.0 if self.d_stop <= 15.0 else 0.0
        dwell_ratio = self.dwell_counter / self.dwell_time
        
        return np.array([d_norm, v_norm, a_norm, ekf_err_norm, is_stop_zone, dwell_ratio], dtype=np.float32)
        
    def step(self, action_idx):
        self.step_count += 1
        
        # Aksiyon ivmesini al
        a_t = self.action_space[action_idx]
        
        # Dinamikler (1D kinematik model)
        # Eğer durakta bekleniyorsa (hız sıfır ve durak toleransındaysa) konum değişmez
        in_stop_and_low_speed = (abs(self.d_stop) <= self.stop_tolerance and self.v <= self.stop_velocity_threshold)
        
        if in_stop_and_low_speed and a_t <= 0.0:
            # Durma durumunu kilitle
            self.v = 0.0
            self.a_prev = 0.0
        else:
            self.d_stop = self.d_stop - self.v * self.dt
            self.v = np.clip(self.v + a_t * self.dt, 0.0, self.max_velocity)
            self.a_prev = a_t
            
        a_diff = abs(a_t - self.a_prev)
        
        # Ödül Hesaplama
        reward = 0.0
        
        # 1. Mesafeye göre ilerleme ödülü (yaklaşma pozitif)
        d_norm = self.d_stop / self.max_distance
        reward -= 0.05 * d_norm  # Durağa ne kadar yakın, o kadar az ceza
        
        # 1b. Sarsıntı cezası (jerk)
        reward -= 0.03 * a_diff
        
        # 2. Durak bölgesine hızlı girme cezası
        if self.d_stop <= 15.0 and self.v > 3.0:
            reward -= 0.2 * (self.v - 3.0)
            
        # 2b. Durağa yaklaşma ödülü (15m içindeyse, hareket ediyorsa)
        if 0.5 < self.d_stop <= 15.0 and self.v > 0.1:
            reward += 0.5  # Devam et, dur değil!
            
        # 3. Adaptif EKF konum hatasına bağlı emniyet cezası
        reward -= 3.0 * self.ekf_error * (self.v / self.max_velocity)
        
        # 3b. Durağa çok uzakta ve hız sıfırsa büyük ceza (early stop)
        if self.d_stop > 5.0 and self.v < 0.1:
            reward -= 2.0  # Her adımda ceza → ajan hareket etmeyi öğrenir
        
        # 4. Durakta bekleme (dwell) mantığı ve ödülleri
        if abs(self.d_stop) <= self.stop_tolerance and self.v <= self.stop_velocity_threshold:
            self.dwell_counter += self.dt
            reward += 8.0  # Doğru durma bölgesinde kalma teşvik ödülü
        else:
            # Bölgeden erken çıkarsa dwell sıfırlanır
            if self.dwell_counter > 0.0 and self.dwell_counter < self.dwell_time:
                reward -= 5.0  # Beklemeyi tamamlamadan hareket etme cezası
            self.dwell_counter = 0.0
            
        info = {"status": "running"}
        
        # Bitiş ve Başarı Koşulları
        
        # A) Bekleme süresi başarıyla tamamlandıysa
        if self.dwell_counter >= self.dwell_time:
            self.done = True
            reward += 150.0
            info["status"] = "success_dwell_complete"
            
        # B) Durağın aşılması (overshoot)
        elif self.d_stop < -1.0:
            self.done = True
            reward -= 100.0
            info["status"] = "overshot"
            
        # C) Maksimum adım sınırı (timeout)
        elif self.step_count >= self.max_steps:
            self.done = True
            # Çok erken durup süre sınırına ulaştıysa
            if self.d_stop > 5.0 and self.v <= 0.1:
                reward -= 50.0
                info["status"] = "early_stop"
            else:
                reward -= 30.0
                info["status"] = "timeout"
                
        return self._get_obs(), reward, self.done, info
