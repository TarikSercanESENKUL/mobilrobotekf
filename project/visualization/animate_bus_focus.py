import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

# project klasörünü python yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.main import run_simulation
from project.visualization.animation_utils import (
    draw_bus_body, draw_route_corridor, draw_stop_zone, save_animation
)
from project.simulation.route_generator import RouteGenerator, StopState

def main():
    print("Otobüs odaklı takip animasyonu (duraklar ve yolcular dahil) hazırlanıyor...")
    res, history = run_simulation(seed=42, scenario_name='combined', return_history=True)
    
    (time_steps, true_history, gnss_history, odom_history, 
     ekf_history, aekf_history, ukf_history, s_ref, is_degraded_history, 
     is_slipping_history, is_near_stop_history, landmarks, aekf_filter) = history
     
    landmarks = np.array(landmarks)
    true_history = np.array(true_history)
    gnss_history = np.array(gnss_history)
    ekf_history = np.array(ekf_history)
    aekf_history = np.array(aekf_history)
    ukf_history = np.array(ukf_history)
    time_steps = np.array(time_steps)
    s_ref = np.array(s_ref)
    
    # 8 adım atlayarak akıcı takip
    frame_skip = 8
    indices = np.arange(0, len(time_steps), frame_skip)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fdfdfd')
    
    # Rota ve Koridorlar
    rg = RouteGenerator()
    route_points = np.column_stack((true_history[:, 0], true_history[:, 1]))
    
    # Çift şeritli yolu çiz
    draw_route_corridor(ax, route_points)
    
    # Rota Çizgisi
    ax.plot(true_history[:, 0], true_history[:, 1], 'k-', linewidth=1.5, label="Gerçek Rota", zorder=2)
    
    # Landmarklar ve Duraklar
    ax.scatter(landmarks[:, 0], landmarks[:, 1], color='gray', marker='x', s=35, alpha=0.6, label="LiDAR Nirengiler")
    for stop_s in rg.bus_stops:
        x_st, y_st, _, _ = rg.get_pose(stop_s)
        draw_stop_zone(ax, [x_st, y_st], stop_radius=10.0, color='orange', alpha=0.1)
        
    # Kestirim yörünge izleri
    ekf_line, = ax.plot([], [], 'b--', linewidth=1.5, label="Klasik EKF")
    aekf_line, = ax.plot([], [], 'g-', linewidth=2.0, label="Adaptif EKF")
    gnss_scatter = ax.scatter([], [], color='red', s=25, alpha=0.5, label="GNSS Ölçümleri")
    
    # Yolcu Çizimleri
    waiting_scatter = ax.scatter([], [], color='#8a2be2', s=25, edgecolor='black', zorder=3, label="Yolcular (Binen/Bekleyen)")
    alighting_scatter = ax.scatter([], [], color='#ff5722', marker='s', s=25, edgecolor='black', zorder=3, label="Yolcular (İnen)")
    
    # GNSS Sapma Vektörü Çizgisi
    gnss_ray, = ax.plot([], [], color='red', linestyle=':', linewidth=1.2, alpha=0.7, label="GNSS Hata Vektörü")
    
    # LiDAR Tarama Işınları Koleksiyonu
    from matplotlib.collections import LineCollection
    lidar_rays = LineCollection([], colors='#00ff00', linewidths=1.0, linestyles='--', alpha=0.7, label="LiDAR Işınları")
    ax.add_collection(lidar_rays)
    
    # Otobüs gövdesi dikdörtgeni
    bus_rect = draw_bus_body(ax, true_history[0, 0], true_history[0, 1], true_history[0, 2],
                             length=12.27, width=2.54, color='#00e5ff', alpha=0.5, edgecolor='#00838f')
    
    ax.set_title("Otobüs Odaklı Dinamik Takip ve Yolcu Dwell Animasyonu")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    
    # Anlık hata ve zaman bilgisi için metin kutusu
    text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
                       
    plt.tight_layout()
    
    def update(frame):
        k = indices[frame]
        x_t, y_t, theta_t, v_t = true_history[k]
        t = time_steps[k]
        
        # Kamera odağını otobüse kitle
        ax.set_xlim(x_t - 22, x_t + 22)
        ax.set_ylim(y_t - 22, y_t + 22)
        
        # Yörünge izlerini güncelle
        ekf_line.set_data(ekf_history[:k+1, 0], ekf_history[:k+1, 1])
        aekf_line.set_data(aekf_history[:k+1, 0], aekf_history[:k+1, 1])
        
        # GNSS Ölçümlerini güncelle
        gps_pts = gnss_history[:k+1]
        valid_gps = gps_pts[~np.isnan(gps_pts[:, 0])]
        if len(valid_gps) > 0:
            gnss_scatter.set_offsets(valid_gps)
            
        # GNSS Hata Vektör çizgisi
        if not np.isnan(gnss_history[k, 0]):
            gnss_ray.set_data([x_t, gnss_history[k, 0]], [y_t, gnss_history[k, 1]])
        else:
            gnss_ray.set_data([], [])
            
        # LiDAR Işınları
        rays = []
        for lm in landmarks:
            dx = lm[0] - x_t
            dy = lm[1] - y_t
            dist = np.hypot(dx, dy)
            if dist <= 50.0:
                global_angle = np.arctan2(dy, dx)
                bearing = (global_angle - theta_t + np.pi) % (2.0 * np.pi) - np.pi
                if abs(bearing) <= np.pi / 2.0:
                    rays.append([(x_t, y_t), (lm[0], lm[1])])
        lidar_rays.set_segments(rays)
        
        # Yolcuları çiz (Tüm duraklardakileri topla)
        p_history_k = res['passenger_history'][k]
        w_all = []
        al_all = []
        for stop in rg.bus_stops:
            w_all.extend(p_history_k[stop]['waiting'])
            al_all.extend(p_history_k[stop]['alighting'])
            
        if len(w_all) > 0:
            waiting_scatter.set_offsets(w_all)
            waiting_scatter.set_visible(True)
        else:
            waiting_scatter.set_visible(False)
            
        if len(al_all) > 0:
            alighting_scatter.set_offsets(al_all)
            alighting_scatter.set_visible(True)
        else:
            alighting_scatter.set_visible(False)
            
        # Otobüs gövdesini güncelle
        t_trans = matplotlib.transforms.Affine2D().rotate(theta_t).translate(x_t, y_t) + ax.transData
        bus_rect.set_transform(t_trans)
        
        # Hatalar
        err_ekf = np.hypot(ekf_history[k, 0] - x_t, ekf_history[k, 1] - y_t)
        err_aekf = np.hypot(aekf_history[k, 0] - x_t, aekf_history[k, 1] - y_t)
        
        # Durumlar
        s_val = s_ref[k] % 1114.0
        slip_status = "KAYMA VAR!" if (700.0 <= s_val <= 800.0) else "Normal"
        gnss_status = "BOZUK" if (300.0 <= s_val <= 500.0) else "Normal"
        
        stop_state = res['stop_states_ref'][k]
        state_str = "CRUISE (Yol Takibi)"
        if stop_state == StopState.DWELL:
            state_str = "Durakta bekleme: yolcu alma/bindirme"
        elif stop_state == StopState.APPROACH:
            state_str = "Durak Yaklaşma"
        elif stop_state == StopState.BRAKE:
            state_str = "Kontrollü Frenleme"
        elif stop_state == StopState.DEPART:
            state_str = "Duraktan Kalkış"
            
        # Metin kutusu güncellemesi
        text_str = (
            f"Zaman: {t:.1f} s | Referans Konumu: {s_ref[k]:.1f} m\n"
            f"Durum: {state_str} | Hız: {v_t:.2f} m/s\n"
            f"Klasik EKF Hatası: {err_ekf:.2f} m | Adaptif EKF Hatası: {err_aekf:.2f} m\n"
            f"Sensör Durumları:\n"
            f"  Odometri Hız: {odom_history[k]:.2f} m/s ({slip_status})\n"
            f"  GNSS Durumu: {gnss_status} (Aktif LM: {len(rays)})"
        )
        text_box.set_text(text_str)
        
        return ekf_line, aekf_line, gnss_scatter, gnss_ray, lidar_rays, text_box, waiting_scatter, alighting_scatter
        
    anim = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)
    
    # GIF olarak kaydet (Yarı yavaş olması için fps=5 ayarlayalım - 100ms interval ile uyumlu)
    output_path = "project/outputs/animations/bus_focus_tracking.gif"
    save_animation(anim, output_path, fps=5)
    plt.close()
    
if __name__ == "__main__":
    main()
