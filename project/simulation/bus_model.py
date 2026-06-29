import numpy as np

class BusModel:
    """
    Otonom otobüsün 2B kinematik modeli.
    AKIA Ultra LF12 EV elektrikli otobüs teknik özellikleri baz alınmıştır.
    """
    def __init__(self, x0=0.0, y0=0.0, theta0=0.0, v0=0.0, dt=0.1):
        # Durum vektörü: [X, Y, theta, v]^T
        self.state = np.array([x0, y0, theta0, v0], dtype=float)
        
        # AKIA Ultra LF12 EV Parametreleri
        self.length = 12.27       # m (Araç uzunluğu)
        self.width = 2.54         # m (Araç genişliği)
        self.L = 6.0              # m (Dingil mesafesi - wheelbase)
        self.dt = dt
        
        # Otobüs dönüş kısıtları
        self.min_turning_radius = 10.5  # m (Minimum dönüş yarıçapı)
        # R_min = L / tan(phi_max) => phi_max = arctan(L / R_min)
        self.max_steering_angle = np.arctan(self.L / self.min_turning_radius)  # rad (~29.74 deg)
        
        # Kütle Parametreleri (Dinamik yolcu kütlesi)
        self.empty_mass = 12000.0  # kg
        self.passenger_capacity = 80
        self.mass_per_passenger = 70.0 # kg
        self.full_mass = self.empty_mass + (self.passenger_capacity * self.mass_per_passenger) # 17600 kg

    def step(self, steering_angle, acceleration):
        """
        Girdilere göre otobüs durumunu günceller.
        steering_angle: phi (direksiyon açısı, radyan)
        acceleration: a (m/s^2 cinsinden ivme)
        """
        x, y, theta, v = self.state
        
        # Direksiyon açısını otobüsün fiziksel dönüş sınırları ile sınırla (min_turning_radius)
        phi_clipped = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Ackermann kinematiği durum güncellemeleri
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        
        # Açısal hız (yaw rate): dot{theta} = (v / L) * tan(phi)
        yaw_rate = (v / self.L) * np.tan(phi_clipped)
        new_theta = (theta + yaw_rate * self.dt) % (2.0 * np.pi)
        
        # Hız güncelleme (hız negatif olamaz)
        new_v = max(0.0, v + acceleration * self.dt)
        
        self.state = np.array([new_x, new_y, new_theta, new_v])
        return self.state

    def get_state(self):
        return self.state
