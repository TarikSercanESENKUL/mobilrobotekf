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
    draw_bus_body, draw_route_corridor, save_animation
)

def main():
    print("GNSS bozulma bölgesi animasyonu hazırlanıyor...")
    res, history = run_simulation(seed=42, scenario_name='combined', return_history=True)
    
    (time_steps, true_history, gnss_history, odom_history, 
     ekf_history, aekf_history, ukf_history, s_ref, is_degraded_history, 
     is_slipping_history, is_near_stop_history, landmarks, aekf_filter) = history
     
    landmarks = np.array(landmarks)
    true_history = np.array(true_history)
    gnss_history = np.array(gnss_history)
    ekf_history = np.array(ekf_history)
    aekf_history = np.array(aekf_history)
    time_steps = np.array(time_steps)
    s_ref = np.array(s_ref)
    
    # Bozulma bölgesi (s: 300-500 m) indekslerini çıkar
    s_mod = s_ref % 1114.0
    degraded_indices = np.where((s_mod >= 300.0) & (s_mod <= 500.0))[0]
    
    # Görüş alanı sınırları
    xs = true_history[degraded_indices, 0]
    ys = true_history[degraded_indices, 1]
    x_min, x_max = np.min(xs) - 15, np.max(xs) + 15
    y_min, y_max = np.min(ys) - 15, np.max(ys) + 15
    
    # Sadece bozulma bölgesine giren çerçeveleri oynatalım (zaman kazanmak ve detayı göstermek için)
    # Bozulma bölgesine girilmeden biraz önce başlayıp biraz sonrasına kadar oynatalım
    zoom_start = max(0, degraded_indices[0] - 20)
    zoom_end = min(len(time_steps) - 1, degraded_indices[-1] + 20)
    
    # 4 adımda bir atlayarak seyreltelim (zoom'da daha fazla kare gösterelim: akıcı olsun)
    frame_skip = 4
    indices = np.arange(zoom_start, zoom_end, frame_skip)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fdfdfd')
    
    # Rota ve Koridorlar
    route_points = np.column_stack((true_history[:, 0], true_history[:, 1]))
    draw_route_corridor(ax, route_points, corridor_width=3.6, color='black', alpha=0.08)
    
    # Bozulma koridorunu kırmızı arka plan olarak çiz
    ax.plot(true_history[degraded_indices, 0], true_history[degraded_indices, 1], 
            color='red', linewidth=10.0, alpha=0.15, label="GNSS Bozulma Bölgesi (Kanyon)", zorder=1)
            
    # Landmarklar (Görüş alanındakiler)
    visible_lms = [lm for lm in landmarks if x_min <= lm[0] <= x_max and y_min <= lm[1] <= y_max]
    if len(visible_lms) > 0:
        visible_lms = np.array(visible_lms)
        ax.scatter(visible_lms[:, 0], visible_lms[:, 1], color='gray', marker='x', s=30, alpha=0.6, label="Nirengiler")
        
    ax.plot(true_history[:, 0], true_history[:, 1], 'k-', linewidth=1.5, label="Gerçek Rota", zorder=2)
    
    # Kestirim izleri (Trails) ve Sensör Işınları
    ekf_line, = ax.plot([], [], 'b--', linewidth=1.5, label="Klasik EKF")
    aekf_line, = ax.plot([], [], 'g-', linewidth=2.0, label="Adaptif EKF")
    gnss_scatter = ax.scatter([], [], color='red', s=25, alpha=0.6, label="Gürültülü GNSS")
    
    # GNSS Sapma Vektörü Çizgisi (Kırmızı Noktalı)
    gnss_ray, = ax.plot([], [], color='red', linestyle=':', linewidth=1.0, alpha=0.6, label="GNSS Hata Vektörü")
    
    # LiDAR Tarama Işınları Koleksiyonu (Yeşil Kesikli)
    from matplotlib.collections import LineCollection
    lidar_rays = LineCollection([], colors='#00ff00', linewidths=0.8, linestyles='--', alpha=0.6, label="LiDAR Işınları")
    ax.add_collection(lidar_rays)
    
    # Otobüs gövdesi
    bus_rect = draw_bus_body(ax, true_history[zoom_start, 0], true_history[zoom_start, 1], true_history[zoom_start, 2],
                             length=12.27, width=2.54, color='#00e5ff', alpha=0.4, edgecolor='#00838f')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("GNSS Bozulma Bölgesi Yakınlaştırılmış Performans Analizi (s: 300-500 m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("equal")
    
    # Metin kutusu
    text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
                       
    plt.tight_layout()
    
    def update(frame):
        k = indices[frame]
        x_t, y_t, theta_t, v_t = true_history[k]
        t = time_steps[k]
        
        # Yörünge izlerini zoom başlangıcından itibaren çiz
        ekf_line.set_data(ekf_history[zoom_start:k+1, 0], ekf_history[zoom_start:k+1, 1])
        aekf_line.set_data(aekf_history[zoom_start:k+1, 0], aekf_history[zoom_start:k+1, 1])
        
        # GNSS noktalarını çiz
        gps_pts = gnss_history[zoom_start:k+1]
        valid_gps = gps_pts[~np.isnan(gps_pts[:, 0])]
        if len(valid_gps) > 0:
            gnss_scatter.set_offsets(valid_gps)
            
        # GNSS Sapma Vektörünü çiz (Mevcut konum ile GNSS ölçümü arası)
        if not np.isnan(gnss_history[k, 0]):
            gnss_ray.set_data([x_t, gnss_history[k, 0]], [y_t, gnss_history[k, 1]])
        else:
            gnss_ray.set_data([], [])
            
        # LiDAR Görünür Landmark Işınları Hesaplama
        rays = []
        for lm in landmarks:
            dx = lm[0] - x_t
            dy = lm[1] - y_t
            dist = np.hypot(dx, dy)
            if dist <= 50.0:  # LiDAR Range
                global_angle = np.arctan2(dy, dx)
                bearing = (global_angle - theta_t + np.pi) % (2.0 * np.pi) - np.pi
                if abs(bearing) <= np.pi / 2.0:  # 180 deg FOV
                    rays.append([(x_t, y_t), (lm[0], lm[1])])
        lidar_rays.set_segments(rays)
            
        # Otobüsün transformasyonu
        t_trans = matplotlib.transforms.Affine2D().rotate(theta_t).translate(x_t, y_t) + ax.transData
        bus_rect.set_transform(t_trans)
        
        # Hata hesaplama
        err_ekf = np.hypot(ekf_history[k, 0] - x_t, ekf_history[k, 1] - y_t)
        err_aekf = np.hypot(aekf_history[k, 0] - x_t, aekf_history[k, 1] - y_t)
        
        # Anlık R_gps değerini aekf traces dizisinden çekelim
        gps_idx = min(len(aekf_filter.r_gps_trace) - 1, int(k * len(aekf_filter.r_gps_trace) / len(time_steps)))
        r_gps_val = aekf_filter.r_gps_trace[gps_idx]
        
        # Metin kutusu metni
        text_str = (
            f"Zaman: {t:.1f} s | Rota Konumu: {s_ref[k]:.1f} m\n"
            f"Filtre Konum Hataları:\n"
            f"  Klasik EKF Hatası: {err_ekf:.2f} m\n"
            f"  Adaptif EKF Hatası: {err_aekf:.2f} m\n"
            f"Sensör Durumları:\n"
            f"  Ölçüm Hızı: {odom_history[k]:.2f} m/s\n"
            f"  GNSS Durumu: GNSS BOZULMASI (Aktif LM: {len(rays)})\n"
            f"  Anlık GNSS Gürültü Kovaryansı ($R_{{gps}}$): {r_gps_val:.2f} m$^2$"
        )
        text_box.set_text(text_str)
        
        return ekf_line, aekf_line, gnss_scatter, gnss_ray, lidar_rays, text_box
        
    anim = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)
    
    output_path = "project/outputs/animations/gnss_degradation_zoom.gif"
    save_animation(anim, output_path, fps=10)
    plt.close()
    
if __name__ == "__main__":
    main()
