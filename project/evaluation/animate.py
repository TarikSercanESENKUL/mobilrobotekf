import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # No GUI window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

from project.main import run_simulation

def generate_landmarks(rg):
    landmarks = []
    s_steps = np.arange(0.0, rg.lap_len, 30.0)
    for s in s_steps:
        x, y, theta, _ = rg.get_pose(s)
        offset = 3.0
        lx1 = x - offset * np.sin(theta)
        ly1 = y + offset * np.cos(theta)
        lx2 = x + offset * np.sin(theta)
        ly2 = y - offset * np.cos(theta)
        landmarks.append([lx1, ly1])
        landmarks.append([lx2, ly2])
    return np.array(landmarks)

def main():
    print("="*60)
    print("OTONOM OTOBÜS SİMÜLASYONU GİF ANİMASYONU ÜRETİLİYOR...")
    print("="*60)
    
    # 1. Simülasyon verilerini nominal run (seed 42, combined) ile al
    print("Simülasyon geçmişi yükleniyor...")
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
    
    N = len(time_steps)
    
    # Zaman serisi hataları
    gnss_errors = np.hypot(gnss_history[:, 0] - true_history[:, 0], gnss_history[:, 1] - true_history[:, 1])
    ekf_errors = np.hypot(ekf_history[:, 0] - true_history[:, 0], ekf_history[:, 1] - true_history[:, 1])
    aekf_errors = np.hypot(aekf_history[:, 0] - true_history[:, 0], aekf_history[:, 1] - true_history[:, 1])
    ukf_errors = np.hypot(ukf_history[:, 0] - true_history[:, 0], ukf_history[:, 1] - true_history[:, 1])
    
    # Subsampling (her 10 adımda bir çizerek 140 karelik hızlı ve hafif bir GIF üretelim)
    step = 10
    frame_indices = np.arange(0, N, step)
    
    # 2. Matplotlib Figure & Subplots Kurulumu
    # Sol panel: Global Görünüm, Orta panel: Lokal Zoom, Sağ paneller: Hata ve Kovaryans Grafikleri
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.0], height_ratios=[1.0, 1.0])
    
    ax_global = fig.add_subplot(gs[:, 0])
    ax_local = fig.add_subplot(gs[:, 1])
    ax_error = fig.add_subplot(gs[0, 2])
    ax_cov = fig.add_subplot(gs[1, 2])
    
    # Stil tanımları
    bg_color = '#121212'
    grid_color = '#2a2a2a'
    fig.patch.set_facecolor(bg_color)
    for ax in [ax_global, ax_local, ax_error, ax_cov]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444444')
            
    # --- Global Panel Kurulumu ---
    ax_global.plot(true_history[:, 0], true_history[:, 1], color='gray', linestyle='--', linewidth=1.5, label="Referans Hat")
    ax_global.scatter(landmarks[:, 0], landmarks[:, 1], color='#333333', marker='x', s=20, label="Landmarklar")
    global_true_line, = ax_global.plot([], [], 'y-', linewidth=1.0)
    global_ekf_line, = ax_global.plot([], [], 'b--', linewidth=1.0)
    global_aekf_line, = ax_global.plot([], [], 'g-', linewidth=1.2)
    global_ukf_line, = ax_global.plot([], [], 'm-.', linewidth=1.0)
    
    global_bus_marker = Rectangle((-3.0, -1.27), 12.27, 2.54, linewidth=1.0, edgecolor='cyan', facecolor='#00e5ff', alpha=0.8)
    ax_global.add_patch(global_bus_marker)
    ax_global.set_title("Küresel Rota ve Konum Takibi", fontsize=12, fontweight='bold')
    ax_global.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper right', prop={'size': 8})
    ax_global.axis("equal")
    ax_global.grid(True, color=grid_color)
    
    # --- Lokal Panel Kurulumu ---
    # Zoomed görünüm
    local_path_line, = ax_local.plot(true_history[:, 0], true_history[:, 1], color='gray', linestyle='-', linewidth=2.5, label="Rota Merkez Çizgisi")
    local_ekf_dot, = ax_local.plot([], [], 'bo', markersize=8, label="Klasik EKF")
    local_aekf_dot, = ax_local.plot([], [], 'go', markersize=9, label="Adaptif EKF")
    local_ukf_dot, = ax_local.plot([], [], 'mo', markersize=8, label="UKF")
    local_gps_dot, = ax_local.plot([], [], 'ro', markersize=10, alpha=0.7, label="GNSS Ölçümü")
    
    # LiDAR Işınları için LineCollection benzeri listeler
    lidar_beams = []
    for _ in range(8):  # Max 8 eşzamanlı ışın
        beam, = ax_local.plot([], [], color='yellow', linestyle=':', alpha=0.7, linewidth=1.0)
        lidar_beams.append(beam)
        
    local_bus_body = Rectangle((-3.0, -1.27), 12.27, 2.54, linewidth=2.0, edgecolor='#00e5ff', facecolor='#00e5ff', alpha=0.4, label="Otobüs Gövdesi")
    ax_local.add_patch(local_bus_body)
    ax_local.set_title("Lokal Araç Görünümü ve LiDAR Tarama", fontsize=12, fontweight='bold')
    ax_local.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper right', prop={'size': 8})
    ax_local.grid(True, color=grid_color)
    
    # --- Hata Panel Kurulumu ---
    ax_error.plot(time_steps, ekf_errors, 'b--', linewidth=1.0, label="Klasik EKF")
    ax_error.plot(time_steps, aekf_errors, 'g-', linewidth=1.5, label="Adaptif EKF")
    ax_error.plot(time_steps, ukf_errors, 'm-.', linewidth=1.0, label="UKF")
    error_time_bar = ax_error.axvline(x=0, color='white', linestyle='-', alpha=0.8)
    ax_error.set_title("Zamana Göre Konum Hatası (m)", fontsize=11, fontweight='bold')
    ax_error.set_ylabel("Hata (m)")
    ax_error.grid(True, color=grid_color)
    ax_error.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper left', prop={'size': 8})
    ax_error.set_ylim(0, 3.0)
    
    # --- Kovaryans Panel Kurulumu ---
    t_gps_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_gps_trace))
    t_odom_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_odom_trace))
    t_lidar_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_lidar_trace))
    
    ax_cov.plot(t_gps_r, aekf_filter.r_gps_trace, 'r-', linewidth=1.2, label="$R_{gps}$")
    ax_cov.plot(t_odom_r, aekf_filter.r_odom_trace, 'orange', linewidth=1.2, label="$R_{odom}$")
    ax_cov.plot(t_lidar_r, aekf_filter.r_lidar_trace, 'g-', linewidth=1.2, label="$R_{lidar}$")
    cov_time_bar = ax_cov.axvline(x=0, color='white', linestyle='-', alpha=0.8)
    ax_cov.set_yscale('log')
    ax_cov.set_title("AEKF Ölçüm Gürültüsü Kovaryans Adaptasyonu ($R_k$)", fontsize=11, fontweight='bold')
    ax_cov.set_ylabel("Varyans (Log Ölçek)")
    ax_cov.set_xlabel("Zaman (s)")
    ax_cov.grid(True, which="both", color=grid_color, linestyle='--', alpha=0.5)
    ax_cov.legend(facecolor='#1e1e1e', labelcolor='white', loc='upper left', prop={'size': 8})
    
    plt.tight_layout()
    
    # 3. Güncelleme Fonksiyonu
    def update(frame_idx):
        k = frame_indices[frame_idx]
        t_curr = time_steps[k]
        
        # Gerçek konum bilgileri
        x_t, y_t, theta_t, v_t = true_history[k]
        
        # --- 3.1 Global Panel Güncellemesi ---
        global_true_line.set_data(true_history[:k+1, 0], true_history[:k+1, 1])
        global_ekf_line.set_data(ekf_history[:k+1, 0], ekf_history[:k+1, 1])
        global_aekf_line.set_data(aekf_history[:k+1, 0], aekf_history[:k+1, 1])
        global_ukf_line.set_data(ukf_history[:k+1, 0], ukf_history[:k+1, 1])
        
        # Global otobüs gövdesi
        t_global = transforms.Affine2D().rotate(theta_t).translate(x_t, y_t) + ax_global.transData
        global_bus_marker.set_transform(t_global)
        
        # --- 3.2 Lokal Panel Güncellemesi ---
        # Görüş alanını güncel konuma odakla
        ax_local.set_xlim(x_t - 25.0, x_t + 25.0)
        ax_local.set_ylim(y_t - 25.0, y_t + 25.0)
        
        # Lokal otobüs gövdesi
        t_local = transforms.Affine2D().rotate(theta_t).translate(x_t, y_t) + ax_local.transData
        local_bus_body.set_transform(t_local)
        
        # Filtre noktaları
        local_ekf_dot.set_data([ekf_history[k, 0]], [ekf_history[k, 1]])
        local_aekf_dot.set_data([aekf_history[k, 0]], [aekf_history[k, 1]])
        local_ukf_dot.set_data([ukf_history[k, 0]], [ukf_history[k, 1]])
        
        # GNSS Noktası
        x_gps, y_gps = gnss_history[k]
        if not np.isnan(x_gps):
            local_gps_dot.set_data([x_gps], [y_gps])
        else:
            local_gps_dot.set_data([], [])
            
        # LiDAR Işınları (Mesafe/açı verilerine göre landmark hatlarını çiz)
        # s_current'a göre görünür landmarkları bulalım
        # Basitleştirmek için, o andaki konuma 35m'den yakın landmarkları bulup çizelim
        dists = np.hypot(landmarks[:, 0] - x_t, landmarks[:, 1] - y_t)
        visible_indices = np.where(dists < 35.0)[0]
        
        # Önce tüm ışınları gizle
        for beam in lidar_beams:
            beam.set_data([], [])
            
        # İlk 8 ışını güncelle
        for idx_beam, idx_lm in enumerate(visible_indices[:8]):
            lm_x, lm_y = landmarks[idx_lm]
            lidar_beams[idx_beam].set_data([x_t, lm_x], [y_t, lm_y])
            
        # --- 3.3 Hata ve Kovaryans Zaman Çizgileri ---
        error_time_bar.set_xdata([t_curr])
        cov_time_bar.set_xdata([t_curr])
        
        # Başlığa anlık bilgileri ekle
        is_deg = is_degraded_history[k]
        is_slip = is_slipping_history[k]
        status_str = "DURUM: "
        if is_deg and is_slip:
            status_str += "GNSS Bozulması + Kayma!"
        elif is_deg:
            status_str += "GNSS Sinyal Bozulması (Urban Canyon)"
        elif is_slip:
            status_str += "Tekerlek Kayması (Slip)"
        else:
            status_str += "Normal Şehir İçi Sürüş"
            
        fig.suptitle(f"Otonom Otobüs Durum Kestirimi Simülasyonu - Zaman: {t_curr:.1f} s / 140.0 s\n{status_str}", 
                     color='white', fontsize=14, fontweight='bold')
                     
        return [global_true_line, global_ekf_line, global_aekf_line, global_ukf_line,
                local_ekf_dot, local_aekf_dot, local_ukf_dot, local_gps_dot,
                error_time_bar, cov_time_bar] + lidar_beams
                
    # 4. Animasyonu Kaydet
    print("GIF dosyası derleniyor (PillowWriter ile)...")
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=100, blit=False)
    
    output_dir = "project/outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = f"{output_dir}/simulation_animation.gif"
    
    writer = PillowWriter(fps=10)
    anim.save(gif_path, writer=writer)
    plt.close()
    
    print(f"GİF Animasyonu başarıyla üretildi ve kaydedildi: {gif_path}")
    print("="*60)

if __name__ == "__main__":
    main()
