import numpy as np

def calculate_rmse(errors):
    """
    Hataların RMSE (Root Mean Squared Error) değerini hesaplar.
    """
    return float(np.sqrt(np.mean(np.array(errors)**2)))

def calculate_mae(errors):
    """
    Hataların MAE (Mean Absolute Error) değerini hesaplar.
    """
    return float(np.mean(np.abs(np.array(errors))))

def angle_difference(a, b):
    """
    İki açı arasındaki en kısa mesafeyi hesaplar (-pi, pi aralığında).
    """
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi

def evaluate_performance(true_trajectory, estimated_trajectory, s_ref, is_degraded_flags, is_near_stop_flags):
    """
    Lokalizasyon performansını değerlendirmek için tüm metrikleri hesaplar.
    true_trajectory: list of numpy arrays [X, Y, theta, v]
    estimated_trajectory: list of numpy arrays [X, Y, theta, v, bg, ba]
    s_ref: list of travel distances
    """
    true_trajectory = np.array(true_trajectory)
    estimated_trajectory = np.array(estimated_trajectory)
    s_ref = np.array(s_ref)
    is_degraded_flags = np.array(is_degraded_flags)
    is_near_stop_flags = np.array(is_near_stop_flags)
    
    N = len(true_trajectory)
    
    # 1. Konum Hataları
    dx = estimated_trajectory[:, 0] - true_trajectory[:, 0]
    dy = estimated_trajectory[:, 1] - true_trajectory[:, 1]
    pos_errors = np.hypot(dx, dy)
    
    # 2. Yönelim Hataları (derece cinsinden)
    heading_errors_rad = np.array([
        abs(angle_difference(est[2], true[2]))
        for est, true in zip(estimated_trajectory, true_trajectory)
    ])
    heading_errors_deg = np.rad2deg(heading_errors_rad)
    
    # 3. Hız Hataları
    speed_errors = np.abs(estimated_trajectory[:, 3] - true_trajectory[:, 3])
    
    # Genel metrikler
    pos_rmse = calculate_rmse(pos_errors)
    pos_mae = calculate_mae(pos_errors)
    pos_max = float(np.max(pos_errors))
    
    heading_rmse = calculate_rmse(heading_errors_deg)
    heading_mae = calculate_mae(heading_errors_deg)
    heading_max = float(np.max(heading_errors_deg))
    
    speed_rmse = calculate_rmse(speed_errors)
    speed_mae = calculate_mae(speed_errors)
    
    # Özel bölgeler analizi
    # 1. GNSS Bozulma Bölgesi
    if np.any(is_degraded_flags):
        pos_errors_degraded = pos_errors[is_degraded_flags]
        pos_max_degraded = float(np.max(pos_errors_degraded))
        pos_rmse_degraded = float(np.sqrt(np.mean(pos_errors_degraded**2)))
    else:
        pos_max_degraded = 0.0
        pos_rmse_degraded = 0.0
        
    # 2. Durak Bölgesi Yanal ve Boylamsal Hatalar
    # s_ref yönelim açısına göre lokal koordinat dönüşümü yapıyoruz
    stop_lat_errors = []
    stop_lon_errors = []
    
    for k in range(N):
        if is_near_stop_flags[k]:
            theta_t = true_trajectory[k, 2]
            ex = estimated_trajectory[k, 0] - true_trajectory[k, 0]
            ey = estimated_trajectory[k, 1] - true_trajectory[k, 1]
            
            # Lokal yanal (lateral) ve boylamsal (longitudinal) hata
            e_lon = ex * np.cos(theta_t) + ey * np.sin(theta_t)
            e_lat = -ex * np.sin(theta_t) + ey * np.cos(theta_t)
            
            stop_lon_errors.append(e_lon)
            stop_lat_errors.append(e_lat)
            
    if len(stop_lat_errors) > 0:
        stop_lat_rmse = calculate_rmse(stop_lat_errors)
        stop_lat_mae = calculate_mae(stop_lat_errors)
        stop_lon_rmse = calculate_rmse(stop_lon_errors)
        stop_lon_mae = calculate_mae(stop_lon_errors)
        pos_rmse_stops = calculate_rmse(np.hypot(stop_lon_errors, stop_lat_errors))
    else:
        stop_lat_rmse = 0.0
        stop_lat_mae = 0.0
        stop_lon_rmse = 0.0
        stop_lon_mae = 0.0
        pos_rmse_stops = 0.0
        
    # 3. Rota Koridoru İhlali Metriği
    # Rota genişliği 3.6 m (sınırlar merkezden itibaren +/- 1.8 m)
    # Otobüs genişliği W = 2.54 m (gövde yarı-genişliği = 1.27 m)
    # Otobüs gövdesinin şeritten taşma miktarı: e_lat_tüm = |e_lat| + W/2
    # Eğer e_lat_tüm > 1.8 m ise otobüs şeritten taşmıştır.
    corridor_departures = 0
    for k in range(N):
        theta_t = true_trajectory[k, 2]
        ex = estimated_trajectory[k, 0] - true_trajectory[k, 0]
        ey = estimated_trajectory[k, 1] - true_trajectory[k, 1]
        e_lat = -ex * np.sin(theta_t) + ey * np.cos(theta_t)
        
        bus_width = 2.54
        if abs(e_lat) + (bus_width / 2.0) > 1.8:
            corridor_departures += 1
            
    corridor_departure_rate = float(corridor_departures / N) * 100.0
    
    return {
        "pos_rmse": pos_rmse,
        "pos_mae": pos_mae,
        "pos_max": pos_max,
        "heading_rmse": heading_rmse,
        "heading_mae": heading_mae,
        "heading_max": heading_max,
        "speed_rmse": speed_rmse,
        "speed_mae": speed_mae,
        "pos_max_degraded": pos_max_degraded,
        "pos_rmse_degraded": pos_rmse_degraded,
        "pos_rmse_stops": pos_rmse_stops,
        "stop_lat_rmse": stop_lat_rmse,
        "stop_lat_mae": stop_lat_mae,
        "stop_lon_rmse": stop_lon_rmse,
        "stop_lon_mae": stop_lon_mae,
        "corridor_departure_rate": corridor_departure_rate
    }

def calculate_improvement_percentage(baseline_rmse, proposed_rmse):
    """
    Klasik EKF'ye göre Adaptif EKF'nin iyileşme yüzdesini hesaplar.
    """
    if baseline_rmse == 0.0:
        return 0.0
    return float(((baseline_rmse - proposed_rmse) / baseline_rmse) * 100.0)


# ─── Şehir İçi Otobüs İşletim Metrikleri ──────────────────────────────────

def compute_lane_keeping_metrics(lateral_errors, timestamps, lane_width=3.6):
    """
    Şerit takip metriklerini hesaplar.
    
    Args:
        lateral_errors: list of float — yanal hata serisi (m)
        timestamps: list of float — zaman damgaları (s)
        lane_width: float — şerit genişliği (m)
        
    Returns:
        dict: Şerit takip metrikleri
    """
    errors = np.abs(np.array(lateral_errors))
    half_lane = lane_width / 2.0
    departures = errors > half_lane
    n_total = len(errors)
    n_departures = int(np.sum(departures))
    
    return {
        'mean_lane_center_error_m': float(np.mean(errors)),
        'max_lane_center_error_m': float(np.max(errors)),
        'rms_lane_error_m': float(np.sqrt(np.mean(errors**2))),
        'lane_departure_count': n_departures,
        'lane_keeping_success_rate_percent': float((n_total - n_departures) / max(1, n_total) * 100),
    }


def compute_stop_operation_metrics(stop_events, passenger_stats=None, door_stats=None):
    """
    Durak işletim metriklerini hesaplar.
    
    Args:
        stop_events: dict — her durak için olaylar (dwell_time, stop_error vb.)
        passenger_stats: dict — yolcu istatistikleri
        door_stats: dict — kapı güvenlik istatistikleri
        
    Returns:
        dict: Kapsamlı durak işletim metrikleri
    """
    metrics = {}
    
    if stop_events:
        dwell_times = [ev.get('dwell_time_s', 0) for ev in stop_events.values() if ev.get('dwell_time_s', 0) > 0]
        boarding_times = [ev.get('boarding_time_s', 0) for ev in stop_events.values() if ev.get('boarding_time_s', 0) > 0]
        alighting_times = [ev.get('alighting_time_s', 0) for ev in stop_events.values() if ev.get('alighting_time_s', 0) > 0]
        stop_errors = [ev.get('stop_error_m', 0) for ev in stop_events.values() if ev.get('stop_error_m', 0) >= 0]
        
        metrics['mean_dwell_time_s'] = float(np.mean(dwell_times)) if dwell_times else 0.0
        metrics['mean_boarding_time_s'] = float(np.mean(boarding_times)) if boarding_times else 0.0
        metrics['mean_alighting_time_s'] = float(np.mean(alighting_times)) if alighting_times else 0.0
        metrics['mean_stop_error_m'] = float(np.mean(stop_errors)) if stop_errors else 0.0
        metrics['max_stop_error_m'] = float(np.max(stop_errors)) if stop_errors else 0.0
    
    if passenger_stats:
        metrics.update({
            'total_boarded_passengers': passenger_stats.get('total_boarded', 0),
            'total_alighted_passengers': passenger_stats.get('total_alighted', 0),
            'avg_waiting_passengers_per_stop': passenger_stats.get('avg_waiting_per_stop', 0.0),
        })
    
    if door_stats:
        metrics.update({
            'unsafe_close_attempt_count': door_stats.get('unsafe_close_attempt_count', 0),
            'successful_safe_close_rate_percent': door_stats.get('successful_safe_close_rate_percent', 100.0),
            'passenger_detected_in_door_zone_count': door_stats.get('passenger_detected_in_door_zone_count', 0),
            'door_reopen_count': door_stats.get('door_reopen_count', 0),
        })
    
    return metrics


def compute_all_city_bus_metrics(
    true_trajectory, estimated_trajectory, lateral_errors,
    stop_events=None, passenger_stats=None, door_stats=None,
    s_ref=None, is_degraded_flags=None, is_near_stop_flags=None
):
    """
    Tüm şehir içi otobüs metriklerini tek bir çağrıda hesaplar.
    Ana evaluate_performance + lane + stop + passenger metrikleri.
    
    Returns:
        dict: Kapsamlı metrik sözlüğü
    """
    if is_degraded_flags is None:
        is_degraded_flags = [False] * len(true_trajectory)
    if is_near_stop_flags is None:
        is_near_stop_flags = [False] * len(true_trajectory)
    if s_ref is None:
        s_ref = list(range(len(true_trajectory)))
    
    # 1. Lokalizasyon metrikleri (mevcut)
    loc_metrics = evaluate_performance(
        true_trajectory, estimated_trajectory,
        s_ref, is_degraded_flags, is_near_stop_flags
    )
    
    # 2. Şerit takip metrikleri
    timestamps = list(range(len(lateral_errors)))
    lane_metrics = compute_lane_keeping_metrics(lateral_errors, timestamps)
    
    # 3. Durak işletim metrikleri
    stop_metrics = compute_stop_operation_metrics(stop_events or {}, passenger_stats, door_stats)
    
    # Tümünü birleştir
    all_metrics = {}
    all_metrics.update(loc_metrics)
    all_metrics.update(lane_metrics)
    all_metrics.update(stop_metrics)
    
    return all_metrics
