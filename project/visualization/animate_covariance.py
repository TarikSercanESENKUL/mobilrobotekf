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
    print("Kovaryans adaptasyon animasyonu hazırlanıyor...")
    res, history = run_simulation(seed=42, scenario_name='combined', return_history=True)
    
    (time_steps, true_history, gnss_history, odom_history, 
     ekf_history, aekf_history, ukf_history, s_ref, is_degraded_history, 
     is_slipping_history, is_near_stop_history, landmarks, aekf_filter) = history
     
    true_history = np.array(true_history)
    ekf_history = np.array(ekf_history)
    aekf_history = np.array(aekf_history)
    time_steps = np.array(time_steps)
    s_ref = np.array(s_ref)
    
    # Zaman serisi hataları
    ekf_errors = np.hypot(ekf_history[:, 0] - true_history[:, 0], ekf_history[:, 1] - true_history[:, 1])
    aekf_errors = np.hypot(aekf_history[:, 0] - true_history[:, 0], aekf_history[:, 1] - true_history[:, 1])
    
    # 10 adımda bir seyreltelim (hızlı olması için)
    frame_skip = 10
    indices = np.arange(0, len(time_steps), frame_skip)
    
    fig, (ax_track, ax_cov) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.2, 1.0]})
    fig.patch.set_facecolor('#ffffff')
    
    # --- Üst Panel: Araç Takibi ---
    ax_track.set_facecolor('#fdfdfd')
    route_points = np.column_stack((true_history[:, 0], true_history[:, 1]))
    draw_route_corridor(ax_track, route_points, corridor_width=3.6, color='black', alpha=0.08)
    
    # GNSS ve Slip Koridorları
    s_mod = s_ref % 1114.0
    degraded_indices = np.where((s_mod >= 300.0) & (s_mod <= 500.0))[0]
    ax_track.plot(true_history[degraded_indices, 0], true_history[degraded_indices, 1], 
                 color='red', linewidth=6.0, alpha=0.15, label="GNSS Bozulması (s: 300-500m)")
                 
    slip_indices = np.where((s_mod >= 700.0) & (s_mod <= 800.0))[0]
    ax_track.plot(true_history[slip_indices, 0], true_history[slip_indices, 1], 
                 color='orange', linewidth=6.0, alpha=0.15, label="Tekerlek Kayması (s: 700-800m)")
                 
    ax_track.plot(true_history[:, 0], true_history[:, 1], 'k-', linewidth=1.5, label="Gerçek Rota")
    
    ekf_line, = ax_track.plot([], [], 'b--', linewidth=1.2, label="Klasik EKF")
    aekf_line, = ax_track.plot([], [], 'g-', linewidth=1.5, label="Adaptif EKF")
    
    bus_rect = draw_bus_body(ax_track, true_history[0, 0], true_history[0, 1], true_history[0, 2],
                             length=12.27, width=2.54, color='#00e5ff', alpha=0.5, edgecolor='#00838f')
                             
    ax_track.set_title("Otonom Otobüs Yörünge Takibi", fontsize=11, fontweight='bold')
    ax_track.axis("equal")
    ax_track.grid(True, linestyle=':', alpha=0.5)
    ax_track.legend(loc="upper right", fontsize=8)
    
    # --- Alt Panel: Kovaryans Adaptasyon Değişimi ---
    ax_cov.set_facecolor('#fdfdfd')
    
    # Referans zaman çizgilerini çizelim
    t_gps_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_gps_trace))
    t_odom_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_odom_trace))
    t_lidar_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_lidar_trace))
    
    # Grafiğin temel çizgileri
    ax_cov.plot(t_gps_r, aekf_filter.r_gps_trace, 'r-', linewidth=1.2, label="GNSS Kovaryansı ($R_{gps}$)")
    ax_cov.plot(t_odom_r, aekf_filter.r_odom_trace, 'orange', linewidth=1.2, label="Odometri Kovaryansı ($R_{odom}$)")
    ax_cov.plot(t_lidar_r, aekf_filter.r_lidar_trace, 'g-', linewidth=1.2, label="LiDAR Kovaryansı ($R_{lidar}$)")
    
    # Hata grafikleri (kıyaslama için)
    ax_cov.plot(time_steps, aekf_errors, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, label="AEKF Konum Hatası (m)")
    
    time_bar = ax_cov.axvline(x=0, color='blue', linestyle='-', alpha=0.8)
    
    ax_cov.set_yscale('log')
    ax_cov.set_title("Dinamik Ölçüm Gürültüsü Kovaryansları ($R_k$)", fontsize=11, fontweight='bold')
    ax_cov.set_ylabel("Varyans (Log Ölçek)")
    ax_cov.set_xlabel("Zaman (s)")
    ax_cov.grid(True, which="both", linestyle='--', alpha=0.5)
    ax_cov.legend(loc="upper left", fontsize=8)
    ax_cov.set_xlim(0, time_steps[-1])
    
    plt.tight_layout()
    
    def update(frame):
        k = indices[frame]
        x_t, y_t, theta_t, _ = true_history[k]
        t = time_steps[k]
        
        # Üst Panel Güncellemesi
        ekf_line.set_data(ekf_history[:k+1, 0], ekf_history[:k+1, 1])
        aekf_line.set_data(aekf_history[:k+1, 0], aekf_history[:k+1, 1])
        
        # Otobüs transformasyonu
        t_trans = matplotlib.transforms.Affine2D().rotate(theta_t).translate(x_t, y_t) + ax_track.transData
        bus_rect.set_transform(t_trans)
        
        # Alt Panel Güncellemesi (Zaman gösterge çizgisi)
        time_bar.set_xdata([t])
        
        # Ana başlığa zamanı ekle
        fig.suptitle(f"Kovaryans Adaptasyonu ve Hata Korelasyonu - Zaman: {t:.1f} s", 
                     fontsize=13, fontweight='bold')
                     
        return ekf_line, aekf_line, time_bar
        
    anim = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)
    
    output_path = "project/outputs/animations/covariance_adaptation.gif"
    save_animation(anim, output_path, fps=10)
    plt.close()
    
if __name__ == "__main__":
    main()
