import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# project klasörünü python yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.main import run_simulation
from project.visualization.animation_utils import (
    draw_bus_body, draw_stop_zone, save_animation, draw_route_corridor
)
from project.simulation.route_generator import RouteGenerator, StopState

def main():
    print("Durak yanaşma ve dwell animasyonu hazırlanıyor...")
    res, history = run_simulation(seed=42, scenario_name='combined', return_history=True)
    
    (time_steps, true_history, gnss_history, odom_history, 
     ekf_history, aekf_history, ukf_history, s_ref, is_degraded_history, 
     is_slipping_history, is_near_stop_history, landmarks, aekf_filter) = history
     
    true_history = np.array(true_history)
    ekf_history = np.array(ekf_history)
    aekf_history = np.array(aekf_history)
    time_steps = np.array(time_steps)
    s_ref = np.array(s_ref)
    
    # s=200m civarındaki durağa en yakın indeksi bul
    stop_idx = np.argmin(np.abs(s_ref - 200.0))
    
    # Durakta bekleme 15 saniye (150 adım) sürer.
    # Durağa 6 saniye kala (60 adım) yaklaşıp, 15 saniye bekleyip, 6 saniye kalkıp uzaklaşalım.
    zoom_start = max(0, stop_idx - 60)
    zoom_end = min(len(time_steps) - 1, stop_idx + 210)
    
    frame_skip = 2
    indices = np.arange(zoom_start, zoom_end, frame_skip)
    
    # Durak Koordinatı
    x_stop, y_stop = true_history[stop_idx, 0], true_history[stop_idx, 1]
    
    # Bounding box
    x_min, x_max = x_stop - 25, x_stop + 25
    y_min, y_max = y_stop - 12, y_stop + 12
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fdfdfd')
    
    # Çift şeritli yolu çiz
    route_points = np.column_stack((true_history[:, 0], true_history[:, 1]))
    draw_route_corridor(ax, route_points)
    
    # Rota Çizgisi
    ax.plot(true_history[:, 0], true_history[:, 1], 'k-', linewidth=1.5, label="Gerçek Rota", zorder=2)
    
    # Durak Yanaşma Bölgesi (radius = 10m)
    draw_stop_zone(ax, [x_stop, y_stop], stop_radius=10.0, color='orange', alpha=0.12)
    
    # Yolcu Çizimleri
    waiting_scatter = ax.scatter([], [], color='#8a2be2', s=35, edgecolor='black', zorder=4, label="Yolcular (Binen/Bekleyen)")
    alighting_scatter = ax.scatter([], [], color='#ff5722', marker='s', s=35, edgecolor='black', zorder=4, label="Yolcular (İnen)")
    
    # Kestirim izleri (Trails)
    ekf_trail, = ax.plot([], [], 'b--', linewidth=1.5, label="Klasik EKF")
    aekf_trail, = ax.plot([], [], 'g-', linewidth=2.0, label="Adaptif EKF")
    
    # Hata Çizgileri
    ekf_err_line, = ax.plot([], [], 'r:', linewidth=1.5, label="Klasik EKF Yanal Sapma")
    aekf_err_line, = ax.plot([], [], 'g:', linewidth=1.5, label="Adaptif EKF Yanal Sapma")
    
    # LiDAR Tarama Işınları
    from matplotlib.collections import LineCollection
    lidar_rays = LineCollection([], colors='#00ff00', linewidths=0.8, linestyles='--', alpha=0.5, label="LiDAR Işınları")
    ax.add_collection(lidar_rays)
    
    # Otobüs gövdesi
    bus_rect = draw_bus_body(ax, true_history[zoom_start, 0], true_history[zoom_start, 1], true_history[zoom_start, 2],
                             length=12.27, width=2.54, color='#00e5ff', alpha=0.5, edgecolor='#00838f')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Çift Şeritli Yol ve Yolcu Boarding/Dwell Simülasyonu (s: 200 m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("equal")
    
    # Durak Durum Etiketi
    stop_label = ax.text(0.5, 0.90, "", transform=ax.transAxes, fontsize=12, fontweight='bold',
                         color='orange', horizontalalignment='center',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.8))
                         
    # Sayısal veri metin kutusu
    text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
                       
    plt.tight_layout()
    
    def update(frame):
        k = indices[frame]
        x_t, y_t, theta_t, v_t = true_history[k]
        t = time_steps[k]
        
        # Yörünge izlerini güncelle
        ekf_trail.set_data(ekf_history[zoom_start:k+1, 0], ekf_history[zoom_start:k+1, 1])
        aekf_trail.set_data(aekf_history[zoom_start:k+1, 0], aekf_history[zoom_start:k+1, 1])
        
        # Hata çizgileri
        ekf_err_line.set_data([x_t, ekf_history[k, 0]], [y_t, ekf_history[k, 1]])
        aekf_err_line.set_data([x_t, aekf_history[k, 0]], [y_t, aekf_history[k, 1]])
        
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
        
        # Yolcuları çiz
        p_data = res['passenger_history'][k][200.0]
        w_pts = np.array(p_data['waiting'])
        al_pts = np.array(p_data['alighting'])
        
        if len(w_pts) > 0:
            waiting_scatter.set_offsets(w_pts)
            waiting_scatter.set_visible(True)
        else:
            waiting_scatter.set_visible(False)
            
        if len(al_pts) > 0:
            alighting_scatter.set_offsets(al_pts)
            alighting_scatter.set_visible(True)
        else:
            alighting_scatter.set_visible(False)
            
        # Otobüs transformasyonu
        t_trans = matplotlib.transforms.Affine2D().rotate(theta_t).translate(x_t, y_t) + ax.transData
        bus_rect.set_transform(t_trans)
        
        # Hatalar
        ex_ekf = ekf_history[k, 0] - x_t
        ey_ekf = ekf_history[k, 1] - y_t
        lat_ekf = -ex_ekf * np.sin(theta_t) + ey_ekf * np.cos(theta_t)
        lon_ekf = ex_ekf * np.cos(theta_t) + ey_ekf * np.sin(theta_t)
        
        ex_aekf = aekf_history[k, 0] - x_t
        ey_aekf = aekf_history[k, 1] - y_t
        lat_aekf = -ex_aekf * np.sin(theta_t) + ey_aekf * np.cos(theta_t)
        lon_aekf = ex_aekf * np.cos(theta_t) + ey_aekf * np.sin(theta_t)
        
        # Durum Makinesi Yazısı
        stop_state = res['stop_states_ref'][k]
        door_status = "AÇIK (Yolcu Tespiti)" if stop_state == StopState.DWELL else "KAPALI"
        
        if stop_state == StopState.DWELL:
            stop_label.set_text("Durakta bekleme: yolcu alma/bindirme")
            stop_label.set_color('#2e7d32')
            stop_label.get_bbox_patch().set_edgecolor('#2e7d32')
        elif stop_state == StopState.APPROACH:
            stop_label.set_text("DURAK YANAŞMA: YAVAŞLAMA")
            stop_label.set_color('orange')
            stop_label.get_bbox_patch().set_edgecolor('orange')
        elif stop_state == StopState.BRAKE:
            stop_label.set_text("KONTROLLÜ DURMA")
            stop_label.set_color('#d84315')
            stop_label.get_bbox_patch().set_edgecolor('#d84315')
        elif stop_state == StopState.DEPART:
            stop_label.set_text("DURAKTAN KALKIŞ: HIZLANMA")
            stop_label.set_color('#1565c0')
            stop_label.get_bbox_patch().set_edgecolor('#1565c0')
        else:
            stop_label.set_text("CRUISE YOL TAKİBİ")
            stop_label.set_color('gray')
            stop_label.get_bbox_patch().set_edgecolor('gray')
            
        text_str = (
            f"Zaman: {t:.1f} s | Referans Konumu: {s_ref[k]:.1f} m\n"
            f"Durum: {stop_state.upper()} | Hız: {v_t:.2f} m/s\n"
            f"Kapı Kamerası: {door_status}\n"
            f"Binen Yolcu: {len(w_pts)} | İnen Yolcu: {len(al_pts)}\n"
            f"Klasik EKF Hataları: Yanal={abs(lat_ekf):.2f}m, Boylamsal={abs(lon_ekf):.2f}m\n"
            f"Adaptif EKF Hataları: Yanal={abs(lat_aekf):.2f}m, Boylamsal={abs(lon_aekf):.2f}m"
        )
        text_box.set_text(text_str)
        
        return ekf_trail, aekf_trail, ekf_err_line, aekf_err_line, stop_label, lidar_rays, text_box, waiting_scatter, alighting_scatter
        
    anim = FuncAnimation(fig, update, frames=len(indices), interval=80, blit=False)
    
    # GIF olarak kaydet
    output_path = "project/outputs/animations/bus_stop_dwell.gif"
    save_animation(anim, output_path, fps=15)
    plt.close()
    
if __name__ == "__main__":
    main()
