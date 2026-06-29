"""
Passenger Model
===============
Gerçekçi şehir içi otobüs yolcu biniş/iniş davranışı.

Her durak için:
- num_waiting: Platformda bekleyen yolcular
- num_alighting: İnecek yolcular (araç içinde)
- num_boarding: Binecek yolcular

Dwell time modeli:
  t_dwell = t_base + n_boarding * t_per_board + n_alighting * t_per_alight

Yolcu durumları:
  WAITING → APPROACHING → BOARDING → INSIDE → ALIGHTING → DEPARTING → DEPARTED
"""
import numpy as np
import copy


class PassengerState:
    WAITING = "waiting"           # Platformda bekliyor
    APPROACHING = "approaching"   # Kapıya yaklaşıyor
    BOARDING = "boarding"         # Kapıdan giriyor
    INSIDE = "inside"             # Araç içinde
    ALIGHTING = "alighting"       # Kapıdan çıkıyor
    DEPARTING = "departing"       # Uzaklaşıyor
    DEPARTED = "departed"         # Gitmiş


class Passenger:
    """Tek bir yolcu."""
    def __init__(self, pid, x, y, state=PassengerState.WAITING, 
                 will_board=True, target_x=None, target_y=None):
        self.pid = pid
        self.x = float(x)
        self.y = float(y)
        self.state = state
        self.will_board = will_board      # Binecek mi (True) yoksa inecek mi (False)
        self.target_x = target_x
        self.target_y = target_y
        self.speed = 1.2                   # m/s yürüyüş hızı
        
    def move_toward_target(self, dt, noise_std=0.05):
        """Hedefe doğru yürü."""
        if self.target_x is None or self.target_y is None:
            return
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = np.hypot(dx, dy)
        if dist < 0.3:
            return
        move = min(self.speed * dt, dist)
        self.x += (dx / dist) * move
        self.y += (dy / dist) * move


class StopPassengerScenario:
    """Tek bir durağın yolcu senaryosu."""
    def __init__(self, stop_s, stop_x, stop_y, stop_theta,
                 num_waiting, num_alighting, rng_seed=None):
        self.stop_s = stop_s
        self.stop_x = stop_x
        self.stop_y = stop_y
        self.stop_theta = stop_theta
        self.rng = np.random.default_rng(rng_seed)
        
        # Dwell time parametreleri
        self.t_base = 5.0       # s (kapı açma/kapama)
        self.t_per_board = 1.5  # s / yolcu
        self.t_per_alight = 1.0 # s / yolcu
        
        # Platform normal yönü (sağ taraf — kaldırım)
        nx_platform = np.sin(stop_theta)
        ny_platform = -np.cos(stop_theta)
        
        # Platform merkezi (otobüsün sağ yanında ~3.5m)
        self.platform_x = stop_x + 3.5 * nx_platform
        self.platform_y = stop_y + 3.5 * ny_platform
        
        # Yolcuları oluştur
        self.passengers = []
        self._create_waiting_passengers(num_waiting)
        self._create_alighting_passengers(num_alighting)
        
        # İstatistik
        self.boarded_count = 0
        self.alighted_count = 0
        self.door_reopen_count = 0
        self.boarding_start_time = None
        self.alighting_start_time = None
        self.boarding_done = False
        self.alighting_done = False
        
    def _create_waiting_passengers(self, n):
        """Platform çevresinde dağınık bekleyen yolcular üret."""
        for i in range(n):
            # Platform etrafında dağınık (1.5m std)
            px = self.platform_x + self.rng.normal(0.0, 1.5)
            py = self.platform_y + self.rng.normal(0.0, 1.5)
            p = Passenger(
                pid=len(self.passengers),
                x=px, y=py,
                state=PassengerState.WAITING,
                will_board=True,
                target_x=None,
                target_y=None
            )
            self.passengers.append(p)
    
    def _create_alighting_passengers(self, n):
        """İnecek yolcuları oluştur — başlangıçta araç içinde."""
        for i in range(n):
            p = Passenger(
                pid=len(self.passengers),
                x=self.stop_x,
                y=self.stop_y,
                state=PassengerState.INSIDE,
                will_board=False,
                target_x=None,
                target_y=None
            )
            self.passengers.append(p)
    
    def get_dwell_time(self):
        """Dinamik dwell süresi hesapla."""
        n_board = sum(1 for p in self.passengers 
                      if p.will_board and p.state != PassengerState.DEPARTED)
        n_alight = sum(1 for p in self.passengers 
                       if not p.will_board and p.state == PassengerState.INSIDE)
        return self.t_base + n_board * self.t_per_board + n_alight * self.t_per_alight
    
    def start_alighting(self, door_positions, bus_state):
        """İnecek yolcuları kapıya yönlendir."""
        alighting_passengers = [p for p in self.passengers 
                                 if p.state == PassengerState.INSIDE]
        
        door_pos = door_positions.get('front_door', (self.stop_x, self.stop_y))
        
        for i, p in enumerate(alighting_passengers):
            p.state = PassengerState.ALIGHTING
            p.x = bus_state[0]  # Araç içindeki pozisyondan başlat
            p.y = bus_state[1]
            # Hedef: kapı pozisyonundan biraz uzakta platform
            angle = self.rng.uniform(0, 2 * np.pi)
            p.target_x = self.platform_x + self.rng.normal(0.0, 1.0)
            p.target_y = self.platform_y + self.rng.normal(0.0, 1.0)
            
    def start_boarding(self, door_positions):
        """Bekleyen yolcuları kapıya yönlendir."""
        waiting_passengers = [p for p in self.passengers 
                               if p.state == PassengerState.WAITING]
        
        # Ön ve arka kapı arasında yolcuları dağıt
        door_keys = list(door_positions.keys())
        
        for i, p in enumerate(waiting_passengers):
            p.state = PassengerState.APPROACHING
            # Alternatif kapılar
            door_key = door_keys[i % len(door_keys)]
            door_pos = door_positions[door_key]
            p.target_x = door_pos[0]
            p.target_y = door_pos[1]
    
    def update(self, dt, bus_state, door_positions, door_is_open):
        """
        Tüm yolcuların durumunu güncelle.
        
        Returns:
            all_boarded (bool): Tüm bekleyenler bindi mi
            all_alighted (bool): Tüm inecekler indi mi
        """
        bx, by, _, _ = bus_state
        
        for p in self.passengers:
            if p.state == PassengerState.WAITING:
                pass  # Bekliyor, hareket yok
                
            elif p.state == PassengerState.APPROACHING:
                if door_is_open and p.target_x is not None:
                    p.move_toward_target(dt)
                    # Kapıya yeterince yaklaştıysa bin
                    dist_to_door = np.hypot(p.target_x - p.x, p.target_y - p.y)
                    if dist_to_door < 0.5:
                        p.state = PassengerState.BOARDING
                        
            elif p.state == PassengerState.BOARDING:
                # Bindi — araç içine geç
                p.state = PassengerState.INSIDE
                p.x = bx
                p.y = by
                self.boarded_count += 1
                
            elif p.state == PassengerState.ALIGHTING:
                p.move_toward_target(dt)
                dist_to_target = np.hypot(p.target_x - p.x, p.target_y - p.y)
                if dist_to_target < 0.5:
                    p.state = PassengerState.DEPARTING
                    self.alighted_count += 1
                    
            elif p.state == PassengerState.DEPARTING:
                # Uzaklaşıyor — platform etrafında hedef değiştir
                if p.target_x is not None:
                    p.move_toward_target(dt)
                    dist = np.hypot(p.target_x - p.x, p.target_y - p.y)
                    if dist < 1.0:
                        p.state = PassengerState.DEPARTED
                        
            elif p.state == PassengerState.INSIDE:
                # Araç hareket ederken araçla birlikte git
                if not door_is_open:
                    p.x = bx + self.rng.normal(0, 0.5)
                    p.y = by + self.rng.normal(0, 0.5)
        
        # Tamamlanma kontrolleri
        boarding_passengers = [p for p in self.passengers if p.will_board]
        all_boarded = all(p.state in [PassengerState.INSIDE, PassengerState.DEPARTED] 
                          for p in boarding_passengers)
        
        alighting_passengers = [p for p in self.passengers if not p.will_board]
        all_alighted = all(p.state in [PassengerState.DEPARTING, PassengerState.DEPARTED]
                           for p in alighting_passengers)
        
        return all_boarded, all_alighted
    
    def get_active_positions(self):
        """
        Animasyon için aktif yolcu pozisyonları.
        
        Returns:
            waiting_pts: [(x, y)] bekleyen yolcular
            boarding_pts: [(x, y)] kapıya yaklaşan/binen yolcular
            alighting_pts: [(x, y)] inen/uzaklaşan yolcular
        """
        waiting_pts = []
        boarding_pts = []
        alighting_pts = []
        
        for p in self.passengers:
            if p.state == PassengerState.WAITING:
                waiting_pts.append((p.x, p.y))
            elif p.state in [PassengerState.APPROACHING, PassengerState.BOARDING]:
                boarding_pts.append((p.x, p.y))
            elif p.state in [PassengerState.ALIGHTING, PassengerState.DEPARTING]:
                alighting_pts.append((p.x, p.y))
                
        return waiting_pts, boarding_pts, alighting_pts
    
    def get_door_zone_passengers(self, door_position, radius=2.0):
        """Kapı bölgesindeki aktif yolcu pozisyonları."""
        door_passengers = []
        dx_d, dy_d = door_position
        for p in self.passengers:
            if p.state in [PassengerState.APPROACHING, PassengerState.BOARDING,
                           PassengerState.ALIGHTING]:
                if np.hypot(p.x - dx_d, p.y - dy_d) <= radius:
                    door_passengers.append((p.x, p.y))
        return door_passengers


class PassengerModel:
    """
    Tüm durakların yolcu senaryolarını yöneten ana sınıf.
    """
    
    # Durak bazlı yolcu senaryoları
    STOP_SCENARIOS = {
        200.0: {'waiting': 5, 'alighting': 2},
        600.0: {'waiting': 0, 'alighting': 4},
        950.0: {'waiting': 8, 'alighting': 1},
    }
    
    def __init__(self, route_generator, rng_seed=42):
        self.rg = route_generator
        self.rng = np.random.default_rng(rng_seed)
        
        # Her durak için senaryo oluştur
        self.stop_scenarios = {}
        for stop_s, scenario_cfg in self.STOP_SCENARIOS.items():
            x_st, y_st, theta_st, _ = route_generator.get_pose(stop_s)
            self.stop_scenarios[stop_s] = StopPassengerScenario(
                stop_s=stop_s,
                stop_x=x_st,
                stop_y=y_st,
                stop_theta=theta_st,
                num_waiting=scenario_cfg['waiting'],
                num_alighting=scenario_cfg['alighting'],
                rng_seed=rng_seed + int(stop_s)
            )
        
        # Global sayaçlar
        self.total_boarded = 0
        self.total_alighted = 0
        
    def get_scenario(self, stop_s):
        """Durak senaryosunu döndür."""
        return self.stop_scenarios.get(stop_s, None)
    
    def get_all_passenger_positions(self):
        """
        Tüm duraklardaki aktif yolcu pozisyonları.
        
        Returns:
            dict: {stop_s: (waiting_pts, boarding_pts, alighting_pts)}
        """
        result = {}
        for stop_s, scenario in self.stop_scenarios.items():
            result[stop_s] = scenario.get_active_positions()
        return result
    
    def get_flat_passenger_positions(self):
        """
        Animasyon için düzleştirilmiş tüm yolcu listesi.
        
        Returns:
            waiting_all, boarding_all, alighting_all: Pozisyon listeleri
        """
        waiting_all, boarding_all, alighting_all = [], [], []
        for scenario in self.stop_scenarios.values():
            w, b, a = scenario.get_active_positions()
            waiting_all.extend(w)
            boarding_all.extend(b)
            alighting_all.extend(a)
        return waiting_all, boarding_all, alighting_all
    
    def get_summary_stats(self):
        """Toplam yolcu istatistikleri."""
        total_boarded = sum(s.boarded_count for s in self.stop_scenarios.values())
        total_alighted = sum(s.alighted_count for s in self.stop_scenarios.values())
        
        scenarios = list(self.stop_scenarios.values())
        avg_waiting = np.mean([
            len([p for p in s.passengers if p.will_board]) 
            for s in scenarios
        ]) if scenarios else 0
        
        return {
            'total_boarded': total_boarded,
            'total_alighted': total_alighted,
            'avg_waiting_per_stop': float(avg_waiting),
        }
