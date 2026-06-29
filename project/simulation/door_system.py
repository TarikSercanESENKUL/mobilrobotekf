"""
Door System
===========
Kapı durum makinesi ve güvenlik kontrol sistemi.

Kapı geçiş akışı:
CLOSED → OPENING → OPEN → BOARDING_ALIGHTING → 
DOOR_CLEARANCE_CHECK → CLOSING → CLOSED

Güvenlik koşulları:
- Kapı bölgesinde yolcu varsa CLOSING → WAITING_CLEARANCE
- Yolcu çekilince → CLOSING → CLOSED → kalkış izni
"""
import numpy as np


class DoorState:
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    BOARDING_ALIGHTING = "boarding_alighting"
    WAITING_CLEARANCE = "waiting_clearance"
    CLOSING = "closing"


class DoorSystem:
    """
    Otobüs kapı sistemi durum makinesi.
    
    - 2 kapı: ön (front) ve arka (rear)
    - Her kapı için bağımsız durum makinesi
    - Kapı kamerası ile entegrasyon
    - Güvenli kapama kontrolü
    """
    
    # Kapı geçiş süreleri
    DOOR_OPENING_TIME = 3.0    # s
    DOOR_CLOSING_TIME = 3.0    # s
    CLEARANCE_CHECK_INTERVAL = 1.0  # s
    
    def __init__(self):
        # Her kapı için bağımsız durum
        self.door_states = {
            'front_door': DoorState.CLOSED,
            'rear_door': DoorState.CLOSED,
        }
        self.door_timers = {
            'front_door': 0.0,
            'rear_door': 0.0,
        }
        
        # İstatistikler
        self.unsafe_close_attempts = 0
        self.safe_closes = 0
        self.door_reopen_count = 0
        self.passenger_detected_in_zone_count = 0
        
    def open_doors(self):
        """Her iki kapıyı açmaya başla."""
        for door_name in self.door_states:
            if self.door_states[door_name] == DoorState.CLOSED:
                self.door_states[door_name] = DoorState.OPENING
                self.door_timers[door_name] = 0.0
    
    def request_close(self):
        """Kapı kapama talebi — güvenlik kontrolü devrede."""
        for door_name in self.door_states:
            state = self.door_states[door_name]
            if state in [DoorState.OPEN, DoorState.BOARDING_ALIGHTING]:
                self.door_states[door_name] = DoorState.WAITING_CLEARANCE
                self.door_timers[door_name] = 0.0
    
    def update(self, dt, door_camera, bus_state, active_passengers, dt_global=0.1):
        """
        Kapı durum makinesini güncelle.
        
        Args:
            dt: Zaman adımı
            door_camera: DoorCameraModel instance
            bus_state: Otobüs durumu
            active_passengers: [(x, y)] tüm aktif yolcu pozisyonları
            
        Returns:
            all_doors_closed (bool): Tüm kapılar kapalıysa True
            boarding_active (bool): Biniş/iniş devam ediyorsa True
        """
        boarding_active = False
        
        for door_name in self.door_states:
            state = self.door_states[door_name]
            self.door_timers[door_name] += dt
            
            if state == DoorState.OPENING:
                if self.door_timers[door_name] >= self.DOOR_OPENING_TIME:
                    self.door_states[door_name] = DoorState.OPEN
                    self.door_timers[door_name] = 0.0
                    
            elif state == DoorState.OPEN:
                self.door_states[door_name] = DoorState.BOARDING_ALIGHTING
                boarding_active = True
                
            elif state == DoorState.BOARDING_ALIGHTING:
                boarding_active = True
                
            elif state == DoorState.WAITING_CLEARANCE:
                # Güvenlik kontrolü
                if self.door_timers[door_name] >= self.CLEARANCE_CHECK_INTERVAL:
                    self.door_timers[door_name] = 0.0
                    
                    safe, front_status, rear_status = door_camera.is_safe_to_close(
                        bus_state, active_passengers
                    )
                    
                    if safe:
                        self.door_states[door_name] = DoorState.CLOSING
                        self.safe_closes += 1
                    else:
                        # Henüz güvenli değil
                        self.passenger_detected_in_zone_count += 1
                        
            elif state == DoorState.CLOSING:
                if self.door_timers[door_name] >= self.DOOR_CLOSING_TIME:
                    # Son güvenlik kontrolü
                    safe, _, _ = door_camera.is_safe_to_close(bus_state, active_passengers)
                    if safe:
                        self.door_states[door_name] = DoorState.CLOSED
                        self.door_timers[door_name] = 0.0
                    else:
                        # Güvensiz — tekrar aç
                        self.door_states[door_name] = DoorState.OPEN
                        self.door_timers[door_name] = 0.0
                        self.door_reopen_count += 1
                        self.unsafe_close_attempts += 1
                        
        all_doors_closed = all(
            s == DoorState.CLOSED for s in self.door_states.values()
        )
        
        return all_doors_closed, boarding_active
    
    def is_any_door_open(self):
        """Herhangi bir kapı açık mı?"""
        return any(
            s not in [DoorState.CLOSED, DoorState.CLOSING]
            for s in self.door_states.values()
        )
    
    def reset(self):
        """Tüm kapıları sıfırla."""
        for door_name in self.door_states:
            self.door_states[door_name] = DoorState.CLOSED
            self.door_timers[door_name] = 0.0
    
    def get_stats(self):
        """Kapı güvenlik istatistikleri."""
        total_closes = self.safe_closes + self.unsafe_close_attempts
        safe_rate = (self.safe_closes / max(1, total_closes)) * 100
        
        return {
            'unsafe_close_attempt_count': self.unsafe_close_attempts,
            'successful_safe_close_rate_percent': safe_rate,
            'passenger_detected_in_door_zone_count': self.passenger_detected_in_zone_count,
            'door_reopen_count': self.door_reopen_count,
        }
    
    def get_door_color(self, door_name):
        """Animasyon için kapı rengi."""
        state = self.door_states[door_name]
        colors = {
            DoorState.CLOSED: '#c62828',           # Kırmızı
            DoorState.OPENING: '#f57f17',          # Turuncu
            DoorState.OPEN: '#2e7d32',             # Yeşil
            DoorState.BOARDING_ALIGHTING: '#1565c0',  # Mavi
            DoorState.WAITING_CLEARANCE: '#f9a825',   # Sarı
            DoorState.CLOSING: '#6a1b9a',          # Mor
        }
        return colors.get(state, '#9e9e9e')
