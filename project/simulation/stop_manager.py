"""
Stop Manager
============
Kapsamlı durak işletim yöneticisi.
Yolcu modeli, kapı sistemi ve durum makinesiyle koordineli çalışır.

Genişletilmiş durak geçiş akışı:
CRUISE → APPROACH_STOP → BRAKE_TO_STOP → 
DOOR_OPEN → ALIGHTING → BOARDING → 
DOOR_CLEARANCE_CHECK → DOOR_CLOSE → DEPART_STOP → CRUISE
"""
import numpy as np


class ExtendedStopState:
    """Genişletilmiş durak durum makinesi durumları."""
    CRUISE = "cruise"
    APPROACH_STOP = "approach_stop"
    BRAKE_TO_STOP = "brake_to_stop"
    DOOR_OPEN = "door_open"
    ALIGHTING = "alighting"
    BOARDING = "boarding"
    DOOR_CLEARANCE_CHECK = "door_clearance_check"
    DOOR_CLOSE = "door_close"
    DEPART_STOP = "depart_stop"


class StopEvent:
    """Tek bir durak olayının kaydı."""
    def __init__(self, stop_s):
        self.stop_s = stop_s
        self.approach_time = None
        self.stop_time = None
        self.door_open_time = None
        self.boarding_complete_time = None
        self.alighting_complete_time = None
        self.door_close_time = None
        self.depart_time = None
        
    @property
    def dwell_time(self):
        if self.door_open_time and self.door_close_time:
            return self.door_close_time - self.door_open_time
        return 0.0
    
    @property
    def boarding_time(self):
        if self.door_open_time and self.boarding_complete_time:
            return self.boarding_complete_time - self.door_open_time
        return 0.0
    
    @property
    def alighting_time(self):
        if self.door_open_time and self.alighting_complete_time:
            return self.alighting_complete_time - self.door_open_time
        return 0.0


class StopManager:
    """
    Kapsamlı durak işletim yöneticisi.
    
    RouteGenerator'ın durak mantığını passenger_model ve door_system ile entegre eder.
    Genişletilmiş durum makinesiyle yolcu biniş/iniş olaylarını koordine eder.
    """
    
    def __init__(self, route_generator, passenger_model, door_system, door_camera, dt=0.1):
        self.rg = route_generator
        self.passenger_model = passenger_model
        self.door_system = door_system
        self.door_camera = door_camera
        self.dt = dt
        
        # Durum makinesi
        self.state = ExtendedStopState.CRUISE
        self.active_stop = None
        self.t_elapsed = 0.0  # Mevcut durumda geçen süre
        
        # Her durak için tamamlanma kaydı
        self.stop_completed = {stop: False for stop in route_generator.bus_stops}
        self.stop_events = {}  # stop_s → StopEvent
        
        # Global zaman sayacı
        self.t_global = 0.0
        
        # Durak parametre eşikleri
        self.approach_distance = 30.0       # m — yaklaşma başlangıcı
        self.brake_distance = 15.0          # m — frenleme başlangıcı
        self.stop_tolerance = 0.5           # m — duruş toleransı
        self.stop_velocity_threshold = 0.2  # m/s — duruş hız eşiği
        
        # Metrikleri toplama
        self.metrics = {
            'dwell_times': [],
            'boarding_times': [],
            'alighting_times': [],
            'stop_errors': [],
            'door_reopen_count': 0,
        }
        
    def _get_stop_distance(self, s_current, stop_s):
        """Mevcut konumdan durağa imzalı mesafe."""
        d = stop_s - (s_current % self.rg.lap_len)
        if d < -self.rg.lap_len / 2:
            d += self.rg.lap_len
        elif d > self.rg.lap_len / 2:
            d -= self.rg.lap_len
        return d
    
    def _get_active_passengers_flat(self):
        """Tüm aktif yolcuların düzleştirilmiş pozisyon listesi."""
        all_pos = []
        for stop_s, scenario in self.passenger_model.stop_scenarios.items():
            w, b, a = scenario.get_active_positions()
            all_pos.extend(w)
            all_pos.extend(b)
            all_pos.extend(a)
        return all_pos
    
    def update(self, s_current, v_current, bus_state, current_time):
        """
        Durak yöneticisini güncelle.
        
        Args:
            s_current: Mevcut yol konumu (m)
            v_current: Mevcut hız (m/s)
            bus_state: Otobüs durumu (x, y, theta, v)
            current_time: Global simülasyon zamanı (s)
            
        Returns:
            state (str): Mevcut genişletilmiş durum
            v_target (float): Hedef hız
            active_stop_s: Aktif durak s koordinatı
            departure_clearance (bool): Kalkış izni
        """
        self.t_global = current_time
        self.t_elapsed += self.dt
        departure_clearance = False
        
        # Tüm duraklara mesafe hesapla
        d_stops = {}
        for stop_s in self.rg.bus_stops:
            d_stops[stop_s] = self._get_stop_distance(s_current, stop_s)
        
        # ─── DURUM MAKİNESİ ───────────────────────────────────────────────
        
        if self.state == ExtendedStopState.CRUISE:
            # Yaklaşma kontrolü
            for stop_s, d_st in d_stops.items():
                if 0.0 < d_st <= self.approach_distance and not self.stop_completed[stop_s]:
                    self.state = ExtendedStopState.APPROACH_STOP
                    self.active_stop = stop_s
                    self.t_elapsed = 0.0
                    self.stop_events[stop_s] = StopEvent(stop_s)
                    self.stop_events[stop_s].approach_time = current_time
                    break
            v_target = self._nominal_speed(s_current)
            
        elif self.state == ExtendedStopState.APPROACH_STOP:
            d_st = d_stops.get(self.active_stop, 999.0)
            if d_st <= self.brake_distance:
                self.state = ExtendedStopState.BRAKE_TO_STOP
                self.t_elapsed = 0.0
            elif d_st < 0.0 or d_st > self.approach_distance + 5.0:
                # Durağı ıskaladı
                self.state = ExtendedStopState.CRUISE
                self.stop_completed[self.active_stop] = True
                self.active_stop = None
            v_target = 5.0  # Yaklaşma hızı
            
        elif self.state == ExtendedStopState.BRAKE_TO_STOP:
            d_st = d_stops.get(self.active_stop, 999.0)
            
            if abs(d_st) <= self.stop_tolerance and v_current <= self.stop_velocity_threshold:
                # Durdu — kapı aç
                self.state = ExtendedStopState.DOOR_OPEN
                self.t_elapsed = 0.0
                
                # Kapıları aç
                self.door_system.open_doors()
                
                if self.active_stop in self.stop_events:
                    self.stop_events[self.active_stop].stop_time = current_time
                    self.stop_events[self.active_stop].door_open_time = current_time
                    self.metrics['stop_errors'].append(abs(d_st))
                    
                # Alighting başlat
                door_positions = self.door_camera.get_all_door_positions(bus_state)
                scenario = self.passenger_model.get_scenario(self.active_stop)
                if scenario:
                    scenario.start_alighting(door_positions, bus_state)
                    
            elif d_st < -1.0:
                # Durağı geçtik
                self.state = ExtendedStopState.CRUISE
                self.stop_completed[self.active_stop] = True
                self.active_stop = None
                
            # Kalan mesafeye göre yavaşlama
            v_target = max(0.05, (max(0.0, d_st) / self.brake_distance) * 6.0)
            
        elif self.state == ExtendedStopState.DOOR_OPEN:
            # Kapı açılıyor
            if self.door_system.is_any_door_open():
                self.state = ExtendedStopState.ALIGHTING
                self.t_elapsed = 0.0
                
                # Boarding başlat
                door_positions = self.door_camera.get_all_door_positions(bus_state)
                scenario = self.passenger_model.get_scenario(self.active_stop)
                if scenario:
                    scenario.start_boarding(door_positions)
            v_target = 0.0
            
        elif self.state == ExtendedStopState.ALIGHTING:
            # İniş ve biniş eş zamanlı
            active_passengers = self._get_active_passengers_flat()
            door_positions = self.door_camera.get_all_door_positions(bus_state)
            
            scenario = self.passenger_model.get_scenario(self.active_stop)
            all_boarded, all_alighted = False, False
            
            if scenario:
                door_is_open = self.door_system.is_any_door_open()
                all_boarded, all_alighted = scenario.update(
                    self.dt, bus_state, door_positions, door_is_open
                )
                
                if all_alighted and self.active_stop in self.stop_events:
                    if self.stop_events[self.active_stop].alighting_complete_time is None:
                        self.stop_events[self.active_stop].alighting_complete_time = current_time
                        
            if all_boarded and all_alighted:
                self.state = ExtendedStopState.BOARDING
                self.t_elapsed = 0.0
                if self.active_stop in self.stop_events:
                    if self.stop_events[self.active_stop].boarding_complete_time is None:
                        self.stop_events[self.active_stop].boarding_complete_time = current_time
                        
            v_target = 0.0
            
        elif self.state == ExtendedStopState.BOARDING:
            # Biniş/iniş bitti — clearance kontrolüne geç
            self.state = ExtendedStopState.DOOR_CLEARANCE_CHECK
            self.t_elapsed = 0.0
            self.door_system.request_close()
            v_target = 0.0
            
        elif self.state == ExtendedStopState.DOOR_CLEARANCE_CHECK:
            # Kapı bölgesi temizleme kontrolü
            active_passengers = self._get_active_passengers_flat()
            all_closed, _ = self.door_system.update(
                self.dt, self.door_camera, bus_state, active_passengers
            )
            
            if all_closed:
                self.state = ExtendedStopState.DOOR_CLOSE
                self.t_elapsed = 0.0
                if self.active_stop in self.stop_events:
                    self.stop_events[self.active_stop].door_close_time = current_time
            v_target = 0.0
            
        elif self.state == ExtendedStopState.DOOR_CLOSE:
            # Kapılar kapandı — kalkış
            self.state = ExtendedStopState.DEPART_STOP
            self.t_elapsed = 0.0
            departure_clearance = True
            
            # Metrikleri kaydet
            if self.active_stop in self.stop_events:
                event = self.stop_events[self.active_stop]
                if event.dwell_time > 0:
                    self.metrics['dwell_times'].append(event.dwell_time)
                if event.boarding_time > 0:
                    self.metrics['boarding_times'].append(event.boarding_time)
                if event.alighting_time > 0:
                    self.metrics['alighting_times'].append(event.alighting_time)
            v_target = 0.0
            
        elif self.state == ExtendedStopState.DEPART_STOP:
            d_st = d_stops.get(self.active_stop, 0.0)
            departure_clearance = True
            
            # Durağı yeterince geride bıraktık
            if s_current - self.active_stop > 10.0 or d_st < -10.0:
                self.stop_completed[self.active_stop] = True
                if self.active_stop in self.stop_events:
                    self.stop_events[self.active_stop].depart_time = current_time
                self.state = ExtendedStopState.CRUISE
                self.active_stop = None
                self.t_elapsed = 0.0
                
            v_target = self._nominal_speed(s_current)
            
        else:
            v_target = self._nominal_speed(s_current)
        
        # DWELL sırasında kapı sistemi güncelle
        if self.state in [ExtendedStopState.ALIGHTING, ExtendedStopState.BOARDING]:
            active_passengers = self._get_active_passengers_flat()
            self.door_system.update(
                self.dt, self.door_camera, bus_state, active_passengers
            )
            
        return self.state, v_target, self.active_stop, departure_clearance
    
    def _nominal_speed(self, s_current):
        """Mevcut konuma göre nominal hız."""
        _, _, _, curv = self.rg.get_pose(s_current)
        if curv > 0.0:
            return 4.0   # Virajda
        return 8.0       # Düzlükte
    
    def get_stop_metrics(self):
        """Durak işletim metriklerini döndür."""
        d = self.door_system.get_stats()
        d['mean_dwell_time_s'] = float(np.mean(self.metrics['dwell_times'])) if self.metrics['dwell_times'] else 0.0
        d['mean_boarding_time_s'] = float(np.mean(self.metrics['boarding_times'])) if self.metrics['boarding_times'] else 0.0
        d['mean_alighting_time_s'] = float(np.mean(self.metrics['alighting_times'])) if self.metrics['alighting_times'] else 0.0
        d['stop_success_rate_percent'] = (len(self.metrics['stop_errors']) / max(1, len(self.rg.bus_stops))) * 100
        d['mean_stop_error_m'] = float(np.mean(self.metrics['stop_errors'])) if self.metrics['stop_errors'] else 0.0
        return d
