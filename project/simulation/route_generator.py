import numpy as np

class StopState:
    CRUISE = "cruise"
    APPROACH = "approach_stop"
    BRAKE = "brake_to_stop"
    DWELL = "dwell_at_stop"
    DEPART = "depart_stop"

class RouteGenerator:
    """
    Otonom otobüs için 2B kapalı çevrim referans rotası üretici.
    Durak yaklaşma ve durakta 15 saniye bekleme durum makinesini simüle eder.
    """
    def __init__(self, dt=0.1, loop_length=None):
        self.dt = dt
        # Racetrack parametreleri
        self.straight_len = 400.0
        self.radius = 50.0
        self.semi_circle_len = np.pi * self.radius
        self.lap_len = 2 * self.straight_len + 2 * self.semi_circle_len
        self.loop_length = self.lap_len if loop_length is None else loop_length
        
        # Durak konumları (s-koordinatı)
        self.bus_stops = [200.0, 600.0, 950.0]
        self.dwell_time = 15.0         # saniye (Durakta bekleme süresi)
        self.stop_tolerance = 0.5      # metre (Duruş toleransı)
        self.stop_velocity_threshold = 0.2  # m/s (Duruş hız eşiği)

    def get_pose(self, s):
        """
        s yol mesafesine göre 2B konum (x, y) ve yönelim (theta) döndürür.
        """
        s = s % self.lap_len
        
        # 1. Düzlük: (0, 0) -> (400, 0)
        if s < self.straight_len:
            x = s
            y = 0.0
            theta = 0.0
            curv = 0.0
        # 1. Viraj: (400, 0) -> (400, 100), merkez (400, 50)
        elif s < self.straight_len + self.semi_circle_len:
            s_rel = s - self.straight_len
            phi = s_rel / self.radius  # 0 -> pi
            x = self.straight_len + self.radius * np.sin(phi)
            y = self.radius - self.radius * np.cos(phi)
            theta = phi
            curv = 1.0 / self.radius
        # 2. Düzlük: (400, 100) -> (0, 100)
        elif s < 2 * self.straight_len + self.semi_circle_len:
            s_rel = s - (self.straight_len + self.semi_circle_len)
            x = self.straight_len - s_rel
            y = 2 * self.radius
            theta = np.pi
            curv = 0.0
        # 2. Viraj: (0, 100) -> (0, 0), merkez (0, 50)
        else:
            s_rel = s - (2 * self.straight_len + self.semi_circle_len)
            phi = s_rel / self.radius  # 0 -> pi
            x = -self.radius * np.sin(phi)
            y = self.radius + self.radius * np.cos(phi)
            theta = (np.pi + phi) % (2 * np.pi)
            curv = 1.0 / self.radius
            
        return x, y, theta, curv

    def generate_trajectory(self, total_time=120.0):
        """
        Zamana bağlı referans durumları [x, y, theta, v], konum s, durum makinesini,
        trafik ışığı durumlarını ve yaya geçiş durumlarını üretir.
        """
        N = int(total_time / self.dt)
        time_steps = np.arange(N) * self.dt
        
        # Çıktı dizileri
        x_ref = np.zeros(N)
        y_ref = np.zeros(N)
        theta_ref = np.zeros(N)
        v_ref = np.zeros(N)
        s_ref = np.zeros(N)
        stop_states_ref = []
        tl_states_ref = []
        ped_states_ref = []
        
        # Otobüs hareket durumu kontrolü
        s = 0.0
        v = 0.0
        
        # Durak bekleme durum makinesi değişkenleri
        state_machine = StopState.CRUISE
        active_stop = None
        stop_dwell_counters = {stop: 0.0 for stop in self.bus_stops}
        stop_completed = {stop: False for stop in self.bus_stops}
        
        for k in range(N):
            t_curr = k * self.dt
            
            # ─── 1. Trafik Işığı (s = 400.0) ──────────────────────────────────
            # 30 saniyelik döngü: 12s Yeşil, 3s Sarı, 15s Kırmızı
            cycle_time = t_curr % 30.0
            if cycle_time < 12.0:
                tl_color = "green"
            elif cycle_time < 15.0:
                tl_color = "yellow"
            else:
                tl_color = "red"
                
            d_light = 400.0 - (s % self.lap_len)
            if d_light < -self.lap_len / 2: d_light += self.lap_len
            elif d_light > self.lap_len / 2: d_light -= self.lap_len
            
            tl_stop = False
            if tl_color in ["red", "yellow"] and 0.0 < d_light <= 30.0:
                tl_stop = True
                
            # ─── 2. Yaya Geçidi (s = 800.0) ────────────────────────────────────
            # 65 - 75 saniyeler arasında yaya yola çıksın
            ped_active = (65.0 <= t_curr <= 75.0)
            d_ped = 800.0 - (s % self.lap_len)
            if d_ped < -self.lap_len / 2: d_ped += self.lap_len
            elif d_ped > self.lap_len / 2: d_ped -= self.lap_len
            
            ped_stop = False
            if ped_active and 0.0 < d_ped <= 25.0:
                ped_stop = True
                
            # Duraklara olan mesafeleri kontrol et
            d_stops = {}
            for stop in self.bus_stops:
                d_stop = stop - (s % self.lap_len)
                # Dairesel pist kapalı çevrim düzeltmesi
                if d_stop < -self.lap_len / 2:
                    d_stop += self.lap_len
                elif d_stop > self.lap_len / 2:
                    d_stop -= self.lap_len
                d_stops[stop] = d_stop

            # Durum Makinesi Geçiş Mantığı
            if state_machine == StopState.CRUISE:
                # 30m kala yaklaşma durumuna geç
                for stop, d_st in d_stops.items():
                    if 0.0 < d_st <= 30.0 and not stop_completed[stop]:
                        state_machine = StopState.APPROACH
                        active_stop = stop
                        break
                        
            elif state_machine == StopState.APPROACH:
                d_st = d_stops[active_stop]
                # 15m kala kontrollü frenlemeye geç
                if d_st <= 15.0:
                    state_machine = StopState.BRAKE
                elif d_st < -1.0 or d_st > 35.0:
                    state_machine = StopState.CRUISE
                    active_stop = None
                    
            elif state_machine == StopState.BRAKE:
                d_st = d_stops[active_stop]
                # 2.5m tolerans içinde ve hız 1.2 m/s altındaysa dur
                if abs(d_st) <= 2.5 and v <= 1.2:
                    state_machine = StopState.DWELL
                    s = active_stop  # Konumu tam durak noktasına kilitle
                    v = 0.0
                elif d_st < -2.5: # Durağı ıskaladıysa devam et
                    state_machine = StopState.CRUISE
                    stop_completed[active_stop] = True
                    active_stop = None
                    
            elif state_machine == StopState.DWELL:
                # Durakta bekleme
                stop_dwell_counters[active_stop] += self.dt
                v = 0.0
                s = active_stop
                if stop_dwell_counters[active_stop] >= self.dwell_time:
                    state_machine = StopState.DEPART
                    s = active_stop + 0.1  # Çıkış yapabilmesi için biraz ilerlet
                    
            elif state_machine == StopState.DEPART:
                # Duraktan ayrılış
                d_st = d_stops[active_stop]
                if (s - active_stop > 5.0) or d_st < -5.0 or d_st > 35.0:
                    state_machine = StopState.CRUISE
                    stop_completed[active_stop] = True
                    active_stop = None

            # Hız Profili Belirleme
            x, y, theta, curv = self.get_pose(s)
            
            # Viraj kontrolü
            if curv > 0.0:
                v_nominal = 4.0  # Virajda güvenli hız
            else:
                v_nominal = 8.0  # Düzlükte nominal hız
                
            # Duruma göre hedef hız profilini ata
            if ped_stop:
                # Yaya için frenleme (2.5m kala dur)
                v_target = max(0.0, (max(0.0, d_ped - 2.5) / 22.5) * v_nominal)
            elif tl_stop:
                # Trafik ışığı için frenleme (2.0m kala dur)
                v_target = max(0.0, (max(0.0, d_light - 2.0) / 28.0) * v_nominal)
            elif state_machine == StopState.CRUISE:
                v_target = v_nominal
            elif state_machine == StopState.APPROACH:
                v_target = 5.0  # Yaklaşırken yavaşla
            elif state_machine == StopState.BRAKE:
                d_st = d_stops[active_stop]
                # Kalan mesafeye göre doğrusal yavaşlama
                v_target = max(0.1, (max(0.0, d_st) / 15.0) * v_nominal)
            elif state_machine == StopState.DWELL:
                v_target = 0.0
            elif state_machine == StopState.DEPART:
                v_target = v_nominal
                
            # İvme ve hız geçiş limitleri
            accel_limit = 1.0   # m/s^2
            decel_limit = -1.5  # m/s^2
            
            if state_machine == StopState.DWELL or (tl_stop and d_light <= 2.5) or (ped_stop and d_ped <= 3.0):
                v = 0.0
            else:
                error = v_target - v
                if error > 0:
                    a = min(accel_limit, error / self.dt)
                else:
                    a = max(decel_limit, error / self.dt)
                v = max(0.0, v + a * self.dt)
                s += v * self.dt
                
            # Güncel konumları ata
            x, y, theta, _ = self.get_pose(s)
            
            x_ref[k] = x
            y_ref[k] = y
            theta_ref[k] = theta
            v_ref[k] = v
            s_ref[k] = s
            stop_states_ref.append(state_machine)
            tl_states_ref.append(tl_color)
            ped_states_ref.append(ped_active)
            
        return time_steps, x_ref, y_ref, theta_ref, v_ref, s_ref, stop_states_ref, tl_states_ref, ped_states_ref
