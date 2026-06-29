"""
City Bus EKF-Integrated Animation
===================================
Ana `run_simulation()` fonksiyonundan gerçek Adaptif EKF tahmin verisini çekerek
şehir içi otobüs animasyonu oluşturur.

Otobüsün hareketi:
- Gerçek fizik: bus.step() ile kinematik model
- EKF tahmini: GNSS + IMU + Odometri + LiDAR + Şerit kamerası
- Yolcu sistemi: Biniş/iniş durum makinesi
- Kapı sistemi: Güvenlik clearance
- 6 kamera FOV görselleştirme

Çıktı: project/outputs/animations/bus_city_simulation.gif
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

from project.main import run_simulation
from project.visualization.animation_utils import (
    draw_route_corridor, draw_stop_zone, save_animation
)
from project.simulation.route_generator import RouteGenerator, StopState
from project.simulation.passenger_model import PassengerModel, PassengerState
from project.simulation.door_system import DoorSystem, DoorState
from project.sensors.door_camera import DoorCameraModel
from project.sensors.surround_camera import SurroundCameraModel


# ─── Kamera Renk Paleti ────────────────────────────────────────────────────
CAMERA_COLORS = {
    'front':              ('#00e5ff', 0.18),
    'rear':               ('#7b1fa2', 0.15),
    'left':               ('#f57f17', 0.15),
    'right':              ('#2e7d32', 0.15),
    'front_left_blind':   ('#c62828', 0.12),
    'front_right_blind':  ('#ad1457', 0.12),
}

DOOR_COLORS = {
    DoorState.CLOSED:              '#c62828',
    DoorState.OPENING:             '#f57f17',
    DoorState.OPEN:                '#2e7d32',
    DoorState.BOARDING_ALIGHTING:  '#1565c0',
    DoorState.WAITING_CLEARANCE:   '#f9a825',
    DoorState.CLOSING:             '#6a1b9a',
}

STOP_STATE_LABELS = {
    StopState.CRUISE:   'CRUISE',
    StopState.APPROACH: 'YAKLAŞIYOR',
    StopState.BRAKE:    'FRENLEME',
    StopState.DWELL:    'DURAKTA (DWELL)',
    StopState.DEPART:   'KALKIŞ',
}


def run_ekf_city_simulation(seed=42, total_time=200.0):
    """
    Ana run_simulation() fonksiyonundan gerçek EKF verisi + yolcu bilgisi al.
    """
    print(f"EKF + Yolcu simülasyonu çalıştırılıyor (seed={seed}, t={total_time}s)...")
    
    # Ana simülasyonu çalıştır
    results, history = run_simulation(
        seed=seed, 
        scenario_name='combined', 
        return_history=True
    )
    
    (time_steps, true_history, gnss_history, odom_history,
     ekf_history, aekf_history, ukf_history, s_ref,
     is_degraded_history, is_slipping_history, is_near_stop_history,
     landmarks, aekf_filter) = history
    
    true_history  = np.array(true_history)
    gnss_history  = np.array(gnss_history)
    ekf_history   = np.array(ekf_history)
    aekf_history  = np.array(aekf_history)
    time_steps    = np.array(time_steps)
    s_ref_arr     = np.array(s_ref)
    
    # Rota yolu (gerçek hareket)
    rg = RouteGenerator(dt=0.1)
    _, _, _, _, _, _, stop_states_ref, tl_states_ref, ped_states_ref = rg.generate_trajectory(total_time)
    
    # Durak konumları
    stop_positions_global = {}
    for stop_s in rg.bus_stops:
        x_st, y_st, _, _ = rg.get_pose(stop_s)
        stop_positions_global[stop_s] = (x_st, y_st)
    
    # Yolcu + Kapı + Kamera sistemleri
    route_waypoints = true_history[:, :2]
    passenger_model = PassengerModel(rg, rng_seed=seed)
    door_system = DoorSystem()
    door_camera = DoorCameraModel(rng_seed=seed)
    surround_camera = SurroundCameraModel()
    
    N = len(time_steps)
    dt = 0.1
    
    # Yolcu/Kapı tarihini oluştur
    stop_triggered   = {stop_s: False for stop_s in rg.bus_stops}
    door_close_triggered = {stop_s: False for stop_s in rg.bus_stops}
    pax_waiting_hist = []
    pax_boarding_hist = []
    pax_alighting_hist = []
    door_state_hist = []
    door_pos_hist = []
    fov_poly_hist = []
    
    prev_stop_state = StopState.CRUISE
    
    for k in range(N):
        bus_state_k = true_history[k]
        s_k = s_ref_arr[k]
        rg_state = stop_states_ref[k] if k < len(stop_states_ref) else StopState.CRUISE
        
        # En yakın durağı tespit et
        active_stop_s = None
        s_mod = s_k % rg.lap_len
        min_d = 1e6
        for stop_s in rg.bus_stops:
            d = stop_s - s_mod
            if d < -rg.lap_len / 2: d += rg.lap_len
            elif d > rg.lap_len / 2: d -= rg.lap_len
            d = abs(d)
            if d < 30.0:
                if d < min_d:
                    min_d = d
                    active_stop_s = stop_s
        
        # DWELL olayları
        if rg_state == StopState.DWELL and active_stop_s is not None:
            scenario = passenger_model.get_scenario(active_stop_s)
            if scenario and not stop_triggered.get(active_stop_s, False):
                door_system.open_doors()
                door_positions_k = door_camera.get_all_door_positions(bus_state_k)
                scenario.start_alighting(door_positions_k, bus_state_k)
                scenario.start_boarding(door_positions_k)
                stop_triggered[active_stop_s] = True
            if scenario:
                door_positions_k = door_camera.get_all_door_positions(bus_state_k)
                scenario.update(dt, bus_state_k, door_positions_k, True)
        
        # DEPART olayları
        if rg_state == StopState.DEPART and active_stop_s is not None:
            if not door_close_triggered.get(active_stop_s, False):
                door_system.request_close()
                door_close_triggered[active_stop_s] = True
            active_pass = []
            for sc in passenger_model.stop_scenarios.values():
                w, b, a = sc.get_active_positions()
                active_pass.extend(b)
                active_pass.extend(a)
            door_system.update(dt, door_camera, bus_state_k, active_pass)
        
        if rg_state == StopState.CRUISE and prev_stop_state != StopState.CRUISE:
            for dk in door_system.door_states:
                door_system.door_states[dk] = 'closed'
        
        prev_stop_state = rg_state
        
        # Yolcu pozisyonları
        w_pts, b_pts, a_pts = passenger_model.get_flat_passenger_positions()
        
        # Kapı + FOV
        door_pos_k = door_camera.get_all_door_positions(bus_state_k)
        fov_polys_k = surround_camera.get_all_fov_polygons(bus_state_k)
        
        pax_waiting_hist.append(list(w_pts))
        pax_boarding_hist.append(list(b_pts))
        pax_alighting_hist.append(list(a_pts))
        door_state_hist.append(dict(door_system.door_states))
        door_pos_hist.append(door_pos_k)
        fov_poly_hist.append(fov_polys_k)
    
    print(f"Simülasyon tamamlandı. {N} adım.")
    
    return {
        't':              time_steps,
        'true':           true_history,
        'gnss':           gnss_history,
        'ekf':            ekf_history,
        'aekf':           aekf_history,
        's_ref':          s_ref_arr,
        'stop_states':    stop_states_ref,
        'tl_states':      tl_states_ref,
        'ped_states':     ped_states_ref,
        'pax_waiting':    pax_waiting_hist,
        'pax_boarding':   pax_boarding_hist,
        'pax_alighting':  pax_alighting_hist,
        'door_states':    door_state_hist,
        'door_positions': door_pos_hist,
        'fov_polygons':   fov_poly_hist,
        'landmarks':      np.array(landmarks),
        'is_degraded':    is_degraded_history,
        'is_slipping':    is_slipping_history,
        'rg':             rg,
        'stop_positions': stop_positions_global,
    }


def make_ekf_city_animation(frame_skip=8, output_path=None, seed=42):
    """
    Gerçek EKF verisi ile şehir otobüsü animasyonu oluştur.
    """
    data = run_ekf_city_simulation(seed=seed)
    T = len(data['t'])
    frame_list = list(range(0, T, frame_skip))
    rg = data['rg']
    route_waypoints = data['true'][:, :2]
    
    # ─── Figür ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10), facecolor='#0a0a0f')
    
    # Ana harita paneli
    ax_map = fig.add_axes([0.01, 0.01, 0.66, 0.98])
    ax_map.set_facecolor('#0d1117')
    ax_map.set_aspect('equal')
    
    # Telemetri paneli
    ax_tel = fig.add_axes([0.69, 0.01, 0.30, 0.98])
    ax_tel.set_facecolor('#0d1117')
    ax_tel.axis('off')
    
    # ─── Yol Çizimi ──────────────────────────────────────────────────────
    draw_route_corridor(ax_map, route_waypoints, corridor_width=7.2)
    
    # GNSS bozulma bölgesi
    s_mod = data['s_ref'] % rg.lap_len
    deg_idx = np.where((s_mod >= 300) & (s_mod <= 500))[0]
    ax_map.plot(data['true'][deg_idx, 0], data['true'][deg_idx, 1],
                color='#ff1744', linewidth=6, alpha=0.12, zorder=0)
    
    # Tekerlek kayma bölgesi
    slip_idx = np.where((s_mod >= 700) & (s_mod <= 800))[0]
    ax_map.plot(data['true'][slip_idx, 0], data['true'][slip_idx, 1],
                color='#ff6d00', linewidth=6, alpha=0.12, zorder=0)
    
    # Durak bölgeleri
    for stop_s, (sx, sy) in data['stop_positions'].items():
        draw_stop_zone(ax_map, (sx, sy), stop_radius=8.0, color='orange', alpha=0.10)
        ax_map.text(sx + 4, sy + 4, f'Durak {int(stop_s)}m',
                    color='#ff9800', fontsize=7, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])
    
    # Trafik Işığı Direği ve Kutusu (s = 400m -> x = 400.0, y = 0.0)
    ax_map.plot([400.0, 400.0], [-1.8, -4.5], color='#666666', linewidth=2.0, zorder=6)
    tl_bulb = patches.Circle((400.0, -4.5), 1.5, facecolor='red', edgecolor='white', linewidth=0.8, zorder=7)
    ax_map.add_patch(tl_bulb)
    ax_map.text(400.0, -8.0, 'Sinyal (400m)', color='#ffffff', fontsize=7, ha='center', fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])
                 
    # Yaya Geçidi Zebra Çizgileri (s = 800m -> x = 157.08, y = 100.0)
    for offset in [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]:
        ax_map.plot([157.08 + offset, 157.08 + offset], [97.0, 103.0], color='#ffffff', linewidth=1.8, alpha=0.4, zorder=1)
    ped_marker = ax_map.scatter([], [], s=100, c='#ffeb3b', marker='*', edgecolors='black', linewidths=0.8, zorder=10)
    ax_map.text(157.08, 105.0, 'Yaya Geçidi', color='#ffffff', fontsize=7, ha='center', fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])

    # Landmarklar
    lm = data['landmarks']
    ax_map.scatter(lm[:, 0], lm[:, 1], color='#444466', marker='+',
                   s=20, alpha=0.6, zorder=1)
    
    # Eksen sınırları
    xmin, xmax = route_waypoints[:, 0].min() - 30, route_waypoints[:, 0].max() + 30
    ymin, ymax = route_waypoints[:, 1].min() - 30, route_waypoints[:, 1].max() + 30
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_xlabel('X (m)', color='#777777', fontsize=9)
    ax_map.set_ylabel('Y (m)', color='#777777', fontsize=9)
    ax_map.tick_params(colors='#444444', labelsize=7.5)
    for sp in ax_map.spines.values():
        sp.set_color('#1a1a1a')
    ax_map.set_title('Otonom Otobüs — Adaptif EKF Lokalizasyon + Şehir İçi Simülasyon',
                      color='#e0e0e0', fontsize=11, fontweight='bold', pad=8)
    
    # ─── Animasyon Nesneleri ─────────────────────────────────────────────
    # Gerçek iz
    true_trail, = ax_map.plot([], [], color='#ffffff', linewidth=1.0, alpha=0.2, zorder=2)
    # EKF iz
    ekf_trail, = ax_map.plot([], [], color='#1565c0', linewidth=1.2, alpha=0.5,
                               linestyle='--', zorder=3, label='Klasik EKF')
    # AEKF iz
    aekf_trail, = ax_map.plot([], [], color='#2e7d32', linewidth=1.5, alpha=0.8,
                                zorder=4, label='Adaptif EKF')
    # GNSS noktalar
    gnss_sc = ax_map.scatter([], [], s=8, c='#ff1744', alpha=0.25, zorder=2, label='GNSS')
    
    # Efsane
    ax_map.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#333355',
                   labelcolor='#cccccc', fontsize=7.5, framealpha=0.9)
    
    # Yolcu scatter
    sc_waiting   = ax_map.scatter([], [], s=45, c='#ce93d8', marker='o',
                                   edgecolors='#9c27b0', linewidth=0.5, zorder=8)
    sc_boarding  = ax_map.scatter([], [], s=55, c='#81d4fa', marker='^',
                                   edgecolors='#0288d1', linewidth=0.5, zorder=9)
    sc_alighting = ax_map.scatter([], [], s=50, c='#ffcc80', marker='v',
                                   edgecolors='#f57c00', linewidth=0.5, zorder=9)
    
    # Otobüs gövdesi (placeholder)
    bus_container = [None, None]  # [true_body, aekf_body]
    
    # Kapı daireleri
    door_patches = {}
    for door_name in ['front_door', 'rear_door']:
        c = plt.Circle((0, 0), radius=1.2, facecolor='#c62828',
                         edgecolor='white', alpha=0.85, linewidth=1.0, zorder=10)
        ax_map.add_patch(c)
        door_patches[door_name] = c
    
    # Kamera FOV polygonları
    fov_objects = {}
    for cam_name, (color, alpha) in CAMERA_COLORS.items():
        poly = plt.Polygon([[0, 0]], closed=True, facecolor=color,
                            edgecolor=color, alpha=0.0, zorder=4)
        ax_map.add_patch(poly)
        fov_objects[cam_name] = poly
    
    # Zaman etiketi
    time_label = ax_map.text(0.01, 0.98, '', transform=ax_map.transAxes,
                              color='#00e5ff', fontsize=9, va='top', fontfamily='monospace',
                              path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])
    
    # Hata etiket
    err_label = ax_map.text(0.01, 0.94, '', transform=ax_map.transAxes,
                             color='#aaaaaa', fontsize=8, va='top', fontfamily='monospace',
                             path_effects=[pe.withStroke(linewidth=2, foreground='#0d1117')])
    
    true_trail_x, true_trail_y = [], []
    ekf_trail_x, ekf_trail_y = [], []
    aekf_trail_x, aekf_trail_y = [], []
    gnss_pts = []
    
    # ─── Telemetri Nesneleri ─────────────────────────────────────────────
    ax_tel.set_title('Telemetri', color='#e0e0e0', fontsize=10, fontweight='bold', pad=8)
    
    tel_labels = [
        ('T (s)', 0.96), ('Gerçek Konum', 0.91),
        ('AEKF Tahmin', 0.86), ('EKF Konum Hatası', 0.81),
        ('AEKF Konum Hatası', 0.76), ('Hız (km/h)', 0.71),
        ('', 0.66),
        ('━━ SENSÖR DURUMLARI ━━', 0.63),
        ('GNSS', 0.58), ('Odometri', 0.53), ('LiDAR Aktif', 0.48),
        ('', 0.43),
        ('━━ DURAK SİSTEMİ ━━', 0.40),
        ('Durak Durumu', 0.35), ('Kapı Ön', 0.31), ('Kapı Arka', 0.27),
        ('Bekleyen Yolcu', 0.23), ('Binen Yolcu', 0.19), ('İnen Yolcu', 0.15),
        ('', 0.12),
        ('━━ KENTSEL AKTÖRLER ━━', 0.09),
        ('Trafik Işığı (400m)', 0.05), ('Yaya Durumu (800m)', 0.01),
    ]
    
    val_texts = {}
    for label, y_pos in tel_labels:
        is_header = label.startswith('━━') or label == ''
        color = '#ffcc02' if label.startswith('━━') else '#aaaaaa'
        ax_tel.text(0.02, y_pos, label, transform=ax_tel.transAxes,
                    color=color, fontsize=8.5, va='top',
                    fontweight='bold' if label.startswith('━━') else 'normal')
        if not is_header:
            v = ax_tel.text(0.52, y_pos, '', transform=ax_tel.transAxes,
                             color='#ffffff', fontsize=8.5, va='top', fontfamily='monospace')
            val_texts[label] = v
    
    # ─── Update Fonksiyonu ─────────────────────────────────────────────
    def update(frame_idx):
        k = frame_idx
        
        # Veriler
        tx, ty, ttheta, tv = data['true'][k, :4]
        ax_x, ay_x = data['aekf'][k, 0], data['aekf'][k, 1]
        ex_x, ey_x = data['ekf'][k, 0], data['ekf'][k, 1]
        t = data['t'][k]
        rg_state = data['stop_states'][k] if k < len(data['stop_states']) else StopState.CRUISE
        door_states = data['door_states'][k]
        w_pts = data['pax_waiting'][k]
        b_pts = data['pax_boarding'][k]
        a_pts = data['pax_alighting'][k]
        door_pos = data['door_positions'][k]
        fov_polys = data['fov_polygons'][k]
        
        # EKF hataları
        err_ekf  = np.hypot(ex_x - tx, ey_x - ty)
        err_aekf = np.hypot(ax_x - tx, ay_x - ty)
        
        # ─ Otobüs gövdeleri
        if bus_container[0] is not None:
            bus_container[0].remove()
        if bus_container[1] is not None:
            bus_container[1].remove()
        
        # Gerçek gövde (beyaz, ince)
        true_rect = patches.Rectangle((-3.0, -1.27), 12.27, 2.54,
                                       linewidth=1.2, edgecolor='#ffffff',
                                       facecolor='#ffffff', alpha=0.15, zorder=5)
        t_mat = transforms.Affine2D().rotate(ttheta).translate(tx, ty) + ax_map.transData
        true_rect.set_transform(t_mat)
        ax_map.add_patch(true_rect)
        bus_container[0] = true_rect
        
        # AEKF tahmini gövde (renkli)
        bus_color = '#00e5ff'
        if rg_state == StopState.DWELL:
            bus_color = '#1565c0'
        elif rg_state == StopState.BRAKE:
            bus_color = '#f57f17'
        elif rg_state == StopState.APPROACH:
            bus_color = '#f9a825'
        
        aekf_theta = np.arctan2(data['aekf'][k, 1] - data['aekf'][max(0,k-1), 1],
                                  data['aekf'][k, 0] - data['aekf'][max(0,k-1), 0]) if k > 0 else ttheta
        
        aekf_rect = patches.Rectangle((-3.0, -1.27), 12.27, 2.54,
                                        linewidth=1.8, edgecolor=bus_color,
                                        facecolor=bus_color, alpha=0.45, zorder=6)
        a_mat = transforms.Affine2D().rotate(ttheta).translate(ax_x, ay_x) + ax_map.transData
        aekf_rect.set_transform(a_mat)
        ax_map.add_patch(aekf_rect)
        bus_container[1] = aekf_rect
        
        # ─ İzler
        true_trail_x.append(tx); true_trail_y.append(ty)
        ekf_trail_x.append(ex_x); ekf_trail_y.append(ey_x)
        aekf_trail_x.append(ax_x); aekf_trail_y.append(ay_x)
        
        max_trail = 150
        if len(true_trail_x) > max_trail:
            true_trail_x.pop(0); true_trail_y.pop(0)
            ekf_trail_x.pop(0); ekf_trail_y.pop(0)
            aekf_trail_x.pop(0); aekf_trail_y.pop(0)
        
        true_trail.set_data(true_trail_x, true_trail_y)
        ekf_trail.set_data(ekf_trail_x, ekf_trail_y)
        aekf_trail.set_data(aekf_trail_x, aekf_trail_y)
        
        # ─ GNSS noktaları
        gx, gy = data['gnss'][k, 0], data['gnss'][k, 1]
        if not np.isnan(gx):
            gnss_pts.append([gx, gy])
            if len(gnss_pts) > 200:
                gnss_pts.pop(0)
        if gnss_pts:
            gnss_sc.set_offsets(np.array(gnss_pts))
        
        # ─ Kamera FOV
        for cam_name, poly_obj in fov_objects.items():
            pts = fov_polys.get(cam_name, [])
            if len(pts) >= 3:
                poly_obj.set_xy(np.array(pts))
                _, alpha = CAMERA_COLORS[cam_name]
                poly_obj.set_alpha(alpha)
            else:
                poly_obj.set_alpha(0.0)
        
        # ─ Yolcu scatter
        sc_waiting.set_offsets(np.array(w_pts) if w_pts else np.empty((0, 2)))
        sc_boarding.set_offsets(np.array(b_pts) if b_pts else np.empty((0, 2)))
        sc_alighting.set_offsets(np.array(a_pts) if a_pts else np.empty((0, 2)))
        
        # ─ Kapı daireleri
        for door_name, circle in door_patches.items():
            if door_name in door_pos:
                dx, dy = door_pos[door_name]
                circle.center = (dx, dy)
                d_state = door_states.get(door_name, DoorState.CLOSED)
                circle.set_facecolor(DOOR_COLORS.get(d_state, '#9e9e9e'))
        
        # ─ Trafik Işığı ve Yaya
        tl_color = data['tl_states'][k] if k < len(data['tl_states']) else 'green'
        ped_active = data['ped_states'][k] if k < len(data['ped_states']) else False
        
        tl_bulb.set_facecolor(tl_color)
        
        if ped_active:
            # Yaya 65s - 75s arasında karşıdan karşıya geçiyor (y=104 -> y=96)
            if t <= 67.0:
                py = 104.0 - 4.0 * (t - 65.0) / 2.0
            elif t <= 73.0:
                py = 100.0
            else:
                py = 100.0 - 4.0 * (t - 73.0) / 2.0
            ped_marker.set_offsets([[157.08, py]])
        else:
            ped_marker.set_offsets(np.empty((0, 2)))

        # ─ Etiketler
        time_label.set_text(f't = {t:.1f}s')
        err_label.set_text(f'EKF: {err_ekf:.3f}m | AEKF: {err_aekf:.3f}m')
        
        # ─ Telemetri
        gnss_status = 'BOZUK' if data['is_degraded'][k] else 'Normal'
        slip_status = 'KAYMA!' if data['is_slipping'][k] else 'Normal'
        state_label = STOP_STATE_LABELS.get(rg_state, str(rg_state))
        front_door_st = door_states.get('front_door', 'closed')
        rear_door_st  = door_states.get('rear_door', 'closed')
        
        # Yaya durumu metni
        if ped_active:
            ped_status = 'YOL BLOKE!' if 67.0 <= t <= 73.0 else 'YAKLAŞIYOR'
        else:
            ped_status = 'SERBEST'
            
        tv_dict = {
            'T (s)':              f'{t:.1f}',
            'Gerçek Konum':       f'({tx:.1f}, {ty:.1f})',
            'AEKF Tahmin':        f'({ax_x:.1f}, {ay_x:.1f})',
            'EKF Konum Hatası':   f'{err_ekf:.3f} m',
            'AEKF Konum Hatası':  f'{err_aekf:.3f} m',
            'Hız (km/h)':         f'{tv*3.6:.1f}',
            'GNSS':               gnss_status,
            'Odometri':           slip_status,
            'LiDAR Aktif':        'Evet',
            'Durak Durumu':       state_label,
            'Kapı Ön':            front_door_st.upper(),
            'Kapı Arka':          rear_door_st.upper(),
            'Bekleyen Yolcu':     str(len(w_pts)),
            'Binen Yolcu':        str(len(b_pts)),
            'İnen Yolcu':         str(len(a_pts)),
            'Trafik Işığı (400m)': tl_color.upper(),
            'Yaya Durumu (800m)':  ped_status,
        }
        
        door_color_map = {
            'closed': '#c62828', 'opening': '#f57f17', 'open': '#2e7d32',
            'boarding_alighting': '#1565c0', 'waiting_clearance': '#f9a825', 'closing': '#6a1b9a',
        }
        
        for label, txt_obj in val_texts.items():
            val = tv_dict.get(label, '')
            txt_obj.set_text(val)
            if label == 'GNSS':
                txt_obj.set_color('#f44336' if 'BOZUK' in val else '#4caf50')
            elif label == 'Odometri':
                txt_obj.set_color('#f44336' if 'KAYMA' in val else '#4caf50')
            elif label == 'AEKF Konum Hatası':
                txt_obj.set_color('#4caf50' if err_aekf < 0.2 else '#f9a825' if err_aekf < 0.5 else '#f44336')
            elif label == 'EKF Konum Hatası':
                txt_obj.set_color('#2196f3' if err_ekf < 0.3 else '#f9a825' if err_ekf < 0.8 else '#f44336')
            elif label == 'Kapı Ön':
                txt_obj.set_color(door_color_map.get(front_door_st, '#ffffff'))
            elif label == 'Kapı Arka':
                txt_obj.set_color(door_color_map.get(rear_door_st, '#ffffff'))
            elif label == 'Trafik Işığı (400m)':
                txt_obj.set_color('#4caf50' if val == 'GREEN' else '#f9a825' if val == 'YELLOW' else '#f44336')
            elif label == 'Yaya Durumu (800m)':
                txt_obj.set_color('#f44336' if 'BLOKE' in val else '#f9a825' if 'YAKLAŞ' in val else '#4caf50')
            else:
                txt_obj.set_color('#ffffff')
        
        return ([true_trail, ekf_trail, aekf_trail, gnss_sc, sc_waiting, sc_boarding,
                  sc_alighting, time_label, err_label,
                  bus_container[0], bus_container[1]] +
                list(fov_objects.values()) + list(door_patches.values()) +
                list(val_texts.values()))
    
    # ─── Animasyon ───────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=frame_list,
                          interval=80, blit=False)
    
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'outputs', 'animations')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'bus_city_simulation.gif')
    
    print(f"Animasyon kaydediliyor: {output_path}")
    save_animation(anim, output_path, fps=12)
    plt.close(fig)
    
    print("Tamamlandı!")
    return output_path


if __name__ == '__main__':
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'animations'
    )
    os.makedirs(out_dir, exist_ok=True)
    make_ekf_city_animation(frame_skip=8,
                             output_path=os.path.join(out_dir, 'bus_city_simulation.gif'))
