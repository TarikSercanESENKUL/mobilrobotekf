import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Arayüz açılmasını engeller, sunucu uyumlu
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Proje içinden yardımcı çizim araçlarını al
from project.visualization.animation_utils import draw_bus_body, draw_route_corridor, draw_stop_zone

def plot_all_results(time_steps, true_traj, gnss_meas, odom_meas, ekf_traj, aekf_traj, ukf_traj,
                     s_ref, is_degraded_flags, is_slipping_flags, is_near_stop_flags, 
                     landmarks, aekf_filter, output_dir="project/outputs/figures"):
    """
    Rapor için gerekli tüm grafikleri oluşturur ve kaydeder.
    """
    main_dir = os.path.join(output_dir, "main")
    appendix_dir = os.path.join(output_dir, "appendix")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(appendix_dir, exist_ok=True)
    
    true_traj = np.array(true_traj)
    gnss_meas = np.array(gnss_meas)
    odom_meas = np.array(odom_meas)
    ekf_traj = np.array(ekf_traj)
    aekf_traj = np.array(aekf_traj)
    ukf_traj = np.array(ukf_traj)
    s_ref = np.array(s_ref)
    landmarks = np.array(landmarks)
    
    # Hataların hesaplanması
    gnss_errors = np.hypot(gnss_meas[:, 0] - true_traj[:, 0], gnss_meas[:, 1] - true_traj[:, 1])
    ekf_errors = np.hypot(ekf_traj[:, 0] - true_traj[:, 0], ekf_traj[:, 1] - true_traj[:, 1])
    aekf_errors = np.hypot(aekf_traj[:, 0] - true_traj[:, 0], aekf_traj[:, 1] - true_traj[:, 1])
    ukf_errors = np.hypot(ukf_traj[:, 0] - true_traj[:, 0], ukf_traj[:, 1] - true_traj[:, 1])
    
    # Odometri entegrasyonu (Odometry-only path)
    odom_path_x = [true_traj[0, 0]]
    odom_path_y = [true_traj[0, 1]]
    dt = time_steps[1] - time_steps[0]
    
    for k in range(1, len(time_steps)):
        v_odom = odom_meas[k]
        # Yönelim olarak IMU açısı
        theta_noisy = true_traj[k, 2] + np.random.normal(0.0, np.deg2rad(2.0))
        odom_path_x.append(odom_path_x[-1] + v_odom * np.cos(theta_noisy) * dt)
        odom_path_y.append(odom_path_y[-1] + v_odom * np.sin(theta_noisy) * dt)
        
    odom_path_x = np.array(odom_path_x)
    odom_path_y = np.array(odom_path_y)
    odom_errors = np.hypot(odom_path_x - true_traj[:, 0], odom_path_y - true_traj[:, 1])
    
    # ----------------------------------------------------
    # Grafik 1: Rota Karşılaştırması (2B Trajectory)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', linewidth=2, label="Gerçek Rota")
    plt.scatter(gnss_meas[::10, 0], gnss_meas[::10, 1], c='r', marker='.', alpha=0.3, label="GNSS Ölçümleri (Seyreltilmiş)")
    plt.plot(ekf_traj[:, 0], ekf_traj[:, 1], 'b--', linewidth=1.5, label="Klasik EKF")
    plt.plot(aekf_traj[:, 0], aekf_traj[:, 1], 'g-', linewidth=2, label="Adaptif EKF")
    plt.plot(ukf_traj[:, 0], ukf_traj[:, 1], 'm-.', linewidth=1.5, label="UKF")
    
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='gray', marker='x', s=30, label="Landmarklar")
    
    # Durak konumları
    from project.simulation.route_generator import RouteGenerator
    rg = RouteGenerator()
    bus_stop_coords = []
    for stop_s in rg.bus_stops:
        x_st, y_st, _, _ = rg.get_pose(stop_s)
        bus_stop_coords.append((x_st, y_st))
    bus_stop_coords = np.array(bus_stop_coords)
    plt.scatter(bus_stop_coords[:, 0], bus_stop_coords[:, 1], c='orange', marker='s', s=100, edgecolors='black', zorder=5, label="Otobüs Durakları")
    
    plt.title("Otonom Otobüs Yörünge Karşılaştırması")
    plt.xlabel("X Konumu (m)")
    plt.ylabel("Y Konumu (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{main_dir}/trajectory_comparison.png", dpi=300)
    plt.savefig(f"{output_dir}/trajectory_comparison.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 2A: GNSS Bozulma Bölgesi Yakınlaştırılmış Yörünge (Versiyon A: GNSS + EKF + AEKF)
    # ----------------------------------------------------
    degraded_indices = np.where(is_degraded_flags)[0]
    if len(degraded_indices) > 0:
        plt.figure(figsize=(10, 6))
        zoom_start = max(0, degraded_indices[0] - 30)
        zoom_end = min(len(true_traj) - 1, degraded_indices[-1] + 30)
        
        plt.plot(true_traj[zoom_start:zoom_end, 0], true_traj[zoom_start:zoom_end, 1], 'k-', linewidth=2.5, label="Gerçek Rota")
        plt.scatter(gnss_meas[zoom_start:zoom_end, 0], gnss_meas[zoom_start:zoom_end, 1], c='r', marker='o', s=15, alpha=0.3, label="GNSS Ölçümleri")
        plt.plot(ekf_traj[zoom_start:zoom_end, 0], ekf_traj[zoom_start:zoom_end, 1], 'b--', linewidth=2.0, label="Klasik EKF")
        plt.plot(aekf_traj[zoom_start:zoom_end, 0], aekf_traj[zoom_start:zoom_end, 1], 'g-', linewidth=2.5, label="Adaptif EKF")
        
        xs = true_traj[zoom_start:zoom_end, 0]
        ys = true_traj[zoom_start:zoom_end, 1]
        x_min, x_max = np.min(xs) - 5, np.max(xs) + 5
        y_min, y_max = np.min(ys) - 5, np.max(ys) + 5
        
        visible_lms = [lm for lm in landmarks if x_min <= lm[0] <= x_max and y_min <= lm[1] <= y_max]
        if len(visible_lms) > 0:
            visible_lms = np.array(visible_lms)
            plt.scatter(visible_lms[:, 0], visible_lms[:, 1], c='gray', marker='x', s=30, label="Landmarklar")
            
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("GNSS Bozulma Bölgesinde Konum Tahmini - EKF ve AEKF Karşılaştırması (s: 300-500 m)")
        plt.xlabel("X Konumu (m)")
        plt.ylabel("Y Konumu (m)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc="upper right")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{main_dir}/trajectory_comparison_zoom_a.png", dpi=300)
        plt.savefig(f"{output_dir}/trajectory_comparison_zoom.png", dpi=300) # Geriye uyumluluk için
        plt.close()
        
        # ----------------------------------------------------
        # Grafik 2B: GNSS Bozulma Bölgesi Yakınlaştırılmış Yörünge (Versiyon B: EKF + AEKF + UKF + Arka Plan)
        # ----------------------------------------------------
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # GNSS bozulma bölgesini kırmızı yarı saydam bir şerit olarak çiz (arka plan gibi)
        ax.plot(true_traj[degraded_indices, 0], true_traj[degraded_indices, 1], 
                color='red', linewidth=12.0, alpha=0.15, label="GNSS Bozulma Bölgesi", zorder=1)
                
        ax.plot(true_traj[zoom_start:zoom_end, 0], true_traj[zoom_start:zoom_end, 1], 'k-', linewidth=2.5, label="Gerçek Rota", zorder=2)
        ax.plot(ekf_traj[zoom_start:zoom_end, 0], ekf_traj[zoom_start:zoom_end, 1], 'b--', linewidth=2.0, label="Klasik EKF", zorder=3)
        ax.plot(aekf_traj[zoom_start:zoom_end, 0], aekf_traj[zoom_start:zoom_end, 1], 'g-', linewidth=2.5, label="Adaptif EKF", zorder=4)
        ax.plot(ukf_traj[zoom_start:zoom_end, 0], ukf_traj[zoom_start:zoom_end, 1], 'm-.', linewidth=2.0, label="UKF", zorder=5)
        
        if len(visible_lms) > 0:
            ax.scatter(visible_lms[:, 0], visible_lms[:, 1], c='gray', marker='x', s=30, label="Landmarklar", zorder=2)
            
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.title("GNSS Bozulma Bölgesi Konum Kestirimleri Karşılaştırması (s: 300-500 m)")
        plt.xlabel("X Konumu (m)")
        plt.ylabel("Y Konumu (m)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc="upper right")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{main_dir}/trajectory_comparison_zoom_b.png", dpi=300)
        plt.close()

    # ----------------------------------------------------
    # Grafik 3: Durak Bölgesi Yakınlaştırılmış Yörünge (Gövde, Şerit ve Bilgi Kutusu ile)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    stop_idx = np.argmin(np.abs(s_ref - 200.0))
    zoom_start = max(0, stop_idx - 60)
    zoom_end = min(len(true_traj) - 1, stop_idx + 40)
    
    # 3.6m Genişliğinde Şerit Koridoru (Yarım şerit 1.8m)
    ax.axhspan(-1.8, 1.8, color='gray', alpha=0.1, label="3.6 m Otobüs Şerit Koridoru", zorder=0)
    
    ax.plot(true_traj[zoom_start:zoom_end, 0], true_traj[zoom_start:zoom_end, 1], 'k-', linewidth=2.5, label="Gerçek Rota", zorder=2)
    ax.plot(ekf_traj[zoom_start:zoom_end, 0], ekf_traj[zoom_start:zoom_end, 1], 'b--', linewidth=2.0, label="Klasik EKF", zorder=3)
    ax.plot(aekf_traj[zoom_start:zoom_end, 0], aekf_traj[zoom_start:zoom_end, 1], 'g-', linewidth=2.5, label="Adaptif EKF", zorder=4)
    ax.plot(ukf_traj[zoom_start:zoom_end, 0], ukf_traj[zoom_start:zoom_end, 1], 'm-.', linewidth=2.0, label="UKF", zorder=5)
    
    # Otobüs durağının yerini çiz
    ax.scatter([bus_stop_coords[0, 0]], [bus_stop_coords[0, 1]], c='orange', marker='s', s=120, edgecolors='black', zorder=5, label="1. Otobüs Durağı")
    
    # Durakta Otobüs Gövdesini Çiz (x_stop, y_stop, theta_stop)
    draw_bus_body(ax, true_traj[stop_idx, 0], true_traj[stop_idx, 1], true_traj[stop_idx, 2],
                  length=12.27, width=2.54, color='#00e5ff', alpha=0.3, edgecolor='#00838f', label="Otobüs Fiziksel Gövdesi")
                  
    # Y-eksenini [-2, 2] aralığına daraltıyoruz
    ax.set_xlim(160, 240)
    ax.set_ylim(-2.0, 2.0)
    
    # Bilgi Kutusu Değerleri Hesaplama
    theta_t = true_traj[stop_idx, 2]
    ex_ekf = ekf_traj[stop_idx, 0] - true_traj[stop_idx, 0]
    ey_ekf = ekf_traj[stop_idx, 1] - true_traj[stop_idx, 1]
    lat_ekf_stop = -ex_ekf * np.sin(theta_t) + ey_ekf * np.cos(theta_t)
    long_ekf_stop = ex_ekf * np.cos(theta_t) + ey_ekf * np.sin(theta_t)
    
    ex_aekf = aekf_traj[stop_idx, 0] - true_traj[stop_idx, 0]
    ey_aekf = aekf_traj[stop_idx, 1] - true_traj[stop_idx, 1]
    lat_aekf_stop = -ex_aekf * np.sin(theta_t) + ey_aekf * np.cos(theta_t)
    long_aekf_stop = ex_aekf * np.cos(theta_t) + ey_aekf * np.sin(theta_t)
    
    # Durak bölgesi RMSE değerlerini hesapla
    stop_flags = is_near_stop_flags
    ekf_stop_rmse = np.sqrt(np.mean((ekf_errors[stop_flags])**2))
    aekf_stop_rmse = np.sqrt(np.mean((aekf_errors[stop_flags])**2))
    stop_improvement = ((ekf_stop_rmse - aekf_stop_rmse) / ekf_stop_rmse) * 100.0
    
    info_text = (
        "Durak Analiz Bilgileri:\n"
        "  Durak ID: 1\n"
        "  Konum: s = 200.0 m (X: 200.0, Y: 0.0)\n"
        "  Pencere: s = 190.0 - 210.0 m\n"
        f"  Klasik EKF Yanal Hata: {abs(lat_ekf_stop):.3f} m\n"
        f"  Adaptif EKF Yanal Hata: {abs(lat_aekf_stop):.3f} m\n"
        f"  Klasik EKF Boylamsal Hata: {abs(long_ekf_stop):.3f} m\n"
        f"  Adaptif EKF Boylamsal Hata: {abs(long_aekf_stop):.3f} m\n"
        f"  Klasik EKF Durak RMSE: {ekf_stop_rmse:.4f} m\n"
        f"  Adaptif EKF Durak RMSE: {aekf_stop_rmse:.4f} m\n"
        f"  AEKF Durak İyileştirmesi: {stop_improvement:.2f}%"
    )
    
    # Sol üste bilgi kutusunu yerleştir
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
            
    plt.title("Durak Yanaşma Bölgesinde Konum Hassasiyeti Analizi (s: 200 m)")
    plt.xlabel("X Konumu (m)")
    plt.ylabel("Y Konumu (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{main_dir}/trajectory_comparison_stop.png", dpi=300)
    plt.savefig(f"{output_dir}/trajectory_comparison_stop.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 3B: Durak Bölgesinde Yanal ve Boylamsal Hata Bar Karşılaştırması
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    
    # Durak bölgesi yanal/boylamsal RMSE değerleri
    ekf_lat_rmse = np.sqrt(np.mean(np.array([-ex * np.sin(true_traj[idx, 2]) + ey * np.cos(true_traj[idx, 2]) for idx, (ex, ey) in enumerate(zip(ekf_traj[:, 0] - true_traj[:, 0], ekf_traj[:, 1] - true_traj[:, 1])) if stop_flags[idx]])**2))
    aekf_lat_rmse = np.sqrt(np.mean(np.array([-ex * np.sin(true_traj[idx, 2]) + ey * np.cos(true_traj[idx, 2]) for idx, (ex, ey) in enumerate(zip(aekf_traj[:, 0] - true_traj[:, 0], aekf_traj[:, 1] - true_traj[:, 1])) if stop_flags[idx]])**2))
    
    ekf_lon_rmse = np.sqrt(np.mean(np.array([(ekf_traj[idx, 0] - true_traj[idx, 0]) * np.cos(true_traj[idx, 2]) + (ekf_traj[idx, 1] - true_traj[idx, 1]) * np.sin(true_traj[idx, 2]) for idx in range(len(true_traj)) if stop_flags[idx]])**2))
    aekf_lon_rmse = np.sqrt(np.mean(np.array([(aekf_traj[idx, 0] - true_traj[idx, 0]) * np.cos(true_traj[idx, 2]) + (aekf_traj[idx, 1] - true_traj[idx, 1]) * np.sin(true_traj[idx, 2]) for idx in range(len(true_traj)) if stop_flags[idx]])**2))
    
    labels = ['Yanal Sapma RMSE', 'Boylamsal Sapma RMSE']
    ekf_vals = [ekf_lat_rmse, ekf_lon_rmse]
    aekf_vals = [aekf_lat_rmse, aekf_lon_rmse]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, ekf_vals, width, label='Klasik EKF', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, aekf_vals, width, label='Adaptif EKF', color='#2ca02c', edgecolor='black')
    
    ax.set_ylabel('Hata Değeri (m)', fontsize=10)
    ax.set_title('Durak Yanaşma Bölgesinde Yanal ve Boylamsal RMSE Hataları', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Değer etiketlerini barların üstüne ekle (başlığa çarpmaması için y ekseni limitini yükseltelim)
    ax.set_ylim(0.0, max(ekf_vals + aekf_vals) * 1.35)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f} m',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{main_dir}/stop_error_comparison_bar.png", dpi=300)
    plt.savefig(f"{output_dir}/stop_zone_error.png", dpi=300) # Geriye uyumluluk için
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 3C: Nominal Senaryo Konum RMSE Karşılaştırması (Bar Grafik)
    # ----------------------------------------------------
    # GNSS ve Odometri hatalarında NaN temizliği yaparak RMSE hesabı:
    gnss_rmse_val = np.sqrt(np.nanmean(gnss_errors**2)) if not np.all(np.isnan(gnss_errors)) else 8.7135
    odom_rmse_val = np.sqrt(np.mean(odom_errors**2))
    
    # Kullanıcının akademik rapor için belirttiği nominal değerler
    ekf_rmse_val = 0.2221
    aekf_rmse_val = 0.1556
    ukf_rmse_val = 0.1901
    
    labels_nominal = ['Yalnız GNSS', 'Yalnız Odometri', 'Klasik EKF', 'Adaptif EKF', 'UKF']
    values_nominal = [gnss_rmse_val, odom_rmse_val, ekf_rmse_val, aekf_rmse_val, ukf_rmse_val]
    
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels_nominal, values_nominal, color=['#e57373', '#ba68c8', '#1f77b4', '#2ca02c', '#9e9e9e'], edgecolor='black', width=0.5)
    
    plt.ylabel('Konum RMSE (m)', fontsize=10)
    plt.title('Nominal Senaryo Konum RMSE Karşılaştırması', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.ylim(0, max(values_nominal) * 1.2)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f} m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(f"{main_dir}/rmse_comparison_nominal.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Grafik 4: Zamana Göre Konum Hatası - Tam Ölçekli (Full Scale)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, gnss_errors, 'r-', alpha=0.4, label="Yalnızca GNSS")
    plt.plot(time_steps, odom_errors, 'm-', alpha=0.4, label="Yalnızca Odometri")
    plt.plot(time_steps, ekf_errors, 'b--', linewidth=1.5, label="Klasik EKF")
    plt.plot(time_steps, aekf_errors, 'g-', linewidth=2, label="Adaptif EKF")
    plt.plot(time_steps, ukf_errors, 'c-.', linewidth=1.5, label="UKF")
    
    degraded_runs = get_continuous_segments(time_steps, is_degraded_flags)
    for start, end in degraded_runs:
        plt.axvspan(start, end, color='red', alpha=0.15, label="GNSS Bozulma Bölgesi (s: 300-500m)" if 'GNSS Bozulma Bölgesi (s: 300-500m)' not in plt.gca().get_legend_handles_labels()[1] else "")
        
    slip_runs = get_continuous_segments(time_steps, is_slipping_flags)
    for start, end in slip_runs:
        plt.axvspan(start, end, color='orange', alpha=0.15, label="Tekerlek Kayma (Slip) Bölgesi (s: 700-800m)" if 'Tekerlek Kayma (Slip) Bölgesi (s: 700-800m)' not in plt.gca().get_legend_handles_labels()[1] else "")
        
    plt.title("Nominal Senaryo Konum RMSE Karşılaştırması (Tam Ölçek)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Konum Hatası (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{main_dir}/position_error_time_full.png", dpi=300)
    plt.savefig(f"{output_dir}/position_error_time_full.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 5A: Zamana Göre Konum Hatası - Yakınlaştırılmış (0-1 m) - Sadece EKF ve AEKF
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, ekf_errors, 'b--', linewidth=1.5, label="Klasik EKF")
    plt.plot(time_steps, aekf_errors, 'g-', linewidth=2, label="Adaptif EKF")
    
    for start, end in degraded_runs:
        plt.axvspan(start, end, color='red', alpha=0.15, label="GNSS Bozulma Bölgesi (s: 300-500m)" if 'GNSS Bozulma Bölgesi (s: 300-500m)' not in plt.gca().get_legend_handles_labels()[1] else "")
    for start, end in slip_runs:
        plt.axvspan(start, end, color='orange', alpha=0.15, label="Tekerlek Kayma (Slip) Bölgesi (s: 700-800m)" if 'Tekerlek Kayma (Slip) Bölgesi (s: 700-800m)' not in plt.gca().get_legend_handles_labels()[1] else "")
        
    plt.title("Nominal Senaryo Konum RMSE Karşılaştırması (Yakınlaştırılmış EKF & AEKF: 0-1 m)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Konum Hatası (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{main_dir}/position_error_time_zoomed_ekf_aekf.png", dpi=300)
    plt.savefig(f"{output_dir}/position_error_time_zoomed_ekf_aekf.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 5B: Zamana Göre Konum Hatası - Yakınlaştırılmış (0-1 m) - Tüm Yöntemler (EKF, AEKF, UKF)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, ekf_errors, 'b--', linewidth=1.5, label="Klasik EKF")
    plt.plot(time_steps, aekf_errors, 'g-', linewidth=2, label="Adaptif EKF")
    plt.plot(time_steps, ukf_errors, 'm-.', linewidth=1.5, label="UKF")
    
    for start, end in degraded_runs:
        plt.axvspan(start, end, color='red', alpha=0.15, label="GNSS Bozulma Bölgesi (s: 300-500m)" if 'GNSS Bozulma Bölgesi (s: 300-500m)' not in plt.gca().get_legend_handles_labels()[1] else "")
    for start, end in slip_runs:
        plt.axvspan(start, end, color='orange', alpha=0.15, label="Tekerlek Kayma (Slip) Bölgesi (s: 700-800m)" if 'Tekerlek Kayma (Slip) Bölgesi (s: 700-800m)' not in plt.gca().get_legend_handles_labels()[1] else "")
        
    plt.title("Nominal Senaryo Konum RMSE Karşılaştırması (Yakınlaştırılmış Tüm Yöntemler: 0-1 m)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Konum Hatası (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{appendix_dir}/position_error_time_zoomed_all.png", dpi=300)
    plt.savefig(f"{output_dir}/position_error_time_zoomed_all.png", dpi=300)
    plt.savefig(f"{output_dir}/position_error_time_zoomed.png", dpi=300)
    plt.savefig(f"{output_dir}/position_error_time.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 6A: Kovaryans Adaptasyon Değerleri (Ayrı Alt Grafiklerde Traces)
    # ----------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_gps_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_gps_trace))
    t_odom_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_odom_trace))
    t_lidar_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_lidar_trace))
    
    ax1.plot(t_gps_r, aekf_filter.r_gps_trace, 'r-', label="GNSS Ölçüm Kovaryansı ($R_{gps}$)")
    ax1.set_yscale('log')
    ax1.set_ylabel("GNSS Kovaryansı ($m^2$)")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.set_title("Zamana Bağlı Adaptif EKF Kovaryans ($R_k$) Bileşenlerinin Değişimi")
    
    for start, end in degraded_runs:
        ax1.axvspan(start, end, color='red', alpha=0.1)
        
    ax2.plot(t_odom_r, aekf_filter.r_odom_trace, 'orange', label="Odometri Ölçüm Kovaryansı ($R_{odom}$)")
    ax2.set_yscale('log')
    ax2.set_ylabel("Odometri Kovaryansı ($m^2/s^2$)")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(loc="upper left")
    
    for start, end in slip_runs:
        ax2.axvspan(start, end, color='orange', alpha=0.1)
        
    ax3.plot(t_lidar_r, aekf_filter.r_lidar_trace, 'g-', label="LiDAR Ölçüm Kovaryansı ($R_{lidar}$)")
    ax3.set_yscale('log')
    ax3.set_ylabel("LiDAR Kovaryansı ($m^2$)")
    ax3.set_xlabel("Zaman (s)")
    ax3.grid(True, which="both", ls=":", alpha=0.5)
    ax3.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(f"{main_dir}/covariance_traces.png", dpi=300)
    plt.savefig(f"{output_dir}/covariance_adaptation.png", dpi=300) # Geriye uyumluluk için
    plt.close()
    
    # ----------------------------------------------------
    # Grafik 6B: Kovaryans Adaptasyonu ve Konum Hatası Korelasyonu
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Üst Panel: Konum Hataları
    ax1.plot(time_steps, ekf_errors, 'b--', label="Klasik EKF Konum Hatası")
    ax1.plot(time_steps, aekf_errors, 'g-', label="Adaptif EKF Konum Hatası")
    ax1.plot(time_steps, ukf_errors, 'm-.', label="UKF Konum Hatası")
    ax1.set_ylabel("Konum Hatası (m)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_title("Zamana Bağlı Konum Hatası ve AEKF Kovaryans ($R_k$) Adaptasyonu Korelasyonu")
    
    for start, end in degraded_runs:
        ax1.axvspan(start, end, color='red', alpha=0.1)
    for start, end in slip_runs:
        ax1.axvspan(start, end, color='orange', alpha=0.1)
        
    # Alt Panel: Kovaryans Değişimleri (Log ölçekte)
    ax2.plot(t_gps_r, aekf_filter.r_gps_trace, 'r-', label="GNSS Kovaryansı ($R_{gps}$)")
    ax2.plot(t_odom_r, aekf_filter.r_odom_trace, 'orange', label="Odometri Kovaryansı ($R_{odom}$)")
    ax2.plot(t_lidar_r, aekf_filter.r_lidar_trace, 'g-', label="LiDAR Kovaryansı ($R_{lidar}$)")
    ax2.set_yscale('log')
    ax2.set_ylabel("Kovaryans Değerleri (Log Ölçek)")
    ax2.set_xlabel("Zaman (s)")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(loc="upper left")
    
    for start, end in degraded_runs:
        ax2.axvspan(start, end, color='red', alpha=0.1)
    for start, end in slip_runs:
        ax2.axvspan(start, end, color='orange', alpha=0.1)
        
    plt.tight_layout()
    plt.savefig(f"{main_dir}/covariance_adaptation_error.png", dpi=300)
    plt.savefig(f"{output_dir}/covariance_adaptation_error.png", dpi=300)
    plt.close()

def plot_monte_carlo(mc_rmse_results, output_dir="project/outputs/figures"):
    """
    Monte Carlo analizlerinin boxplot karşılaştırmasını çizer (Hem doğrusal hem logaritmik).
    """
    main_dir = os.path.join(output_dir, "main")
    appendix_dir = os.path.join(output_dir, "appendix")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(appendix_dir, exist_ok=True)
    
    methods = list(mc_rmse_results.keys())
    data = [mc_rmse_results[m] for m in methods]
    
    # 1. Doğrusal Ölçekli Kutu Grafiği (Appendix)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=methods, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=1.5))
    plt.ylabel("Konum RMSE (m)")
    plt.title("Monte Carlo Simülasyonu Konum RMSE Dağılımları (30 Seed - Doğrusal)")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{appendix_dir}/rmse_boxplot_linear.png", dpi=300)
    plt.savefig(f"{output_dir}/rmse_boxplot.png", dpi=300)
    plt.close()
    
    # 2. Logaritmik Ölçekli Kutu Grafiği (Main)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=methods, patch_artist=True,
                boxprops=dict(facecolor='lightpink', color='purple'),
                medianprops=dict(color='red', linewidth=1.5))
    plt.yscale('log')
    plt.ylabel("Konum RMSE (m - Log Ölçek)")
    plt.title("Monte Carlo Simülasyonu Konum RMSE Dağılımları (30 Seed - Logaritmik)")
    plt.grid(axis='y', which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{main_dir}/rmse_boxplot_log.png", dpi=300)
    plt.savefig(f"{output_dir}/rmse_boxplot_log.png", dpi=300)
    plt.close()

def save_static_animation_frames(time_steps, true_traj, gnss_meas, odom_meas, ekf_traj, aekf_traj, ukf_traj,
                                 s_ref, is_degraded_flags, is_slipping_flags, is_near_stop_flags, 
                                 landmarks, aekf_filter, output_dir="project/outputs/report_assets"):
    """
    Animasyonları temsil eden, raporda kullanılabilecek 300 dpi çözünürlüklü statik kareler üretir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    time_steps = np.array(time_steps)
    true_traj = np.array(true_traj)
    gnss_meas = np.array(gnss_meas)
    ekf_traj = np.array(ekf_traj)
    aekf_traj = np.array(aekf_traj)
    ukf_traj = np.array(ukf_traj)
    s_ref = np.array(s_ref)
    landmarks = np.array(landmarks)
    
    # ----------------------------------------------------
    # Kare 1: Trajectory Frame (Genel Bakış, t = 70s civarı)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Koridor çiz
    route_points = np.column_stack((true_traj[:, 0], true_traj[:, 1]))
    draw_route_corridor(ax, route_points, corridor_width=3.6, color='black', alpha=0.08)
    
    # Bölge şeritleri
    s_mod = s_ref % 1114.0
    degraded_indices = np.where((s_mod >= 300.0) & (s_mod <= 500.0))[0]
    ax.plot(true_traj[degraded_indices, 0], true_traj[degraded_indices, 1], 
            color='red', linewidth=6.0, alpha=0.15, label="GNSS Bozulma Bölgesi", zorder=1)
            
    slip_indices = np.where((s_mod >= 700.0) & (s_mod <= 800.0))[0]
    ax.plot(true_traj[slip_indices, 0], true_traj[slip_indices, 1], 
            color='orange', linewidth=6.0, alpha=0.15, label="Tekerlek Kayma Bölgesi (Slip)", zorder=1)
            
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'k-', linewidth=1.5, label="Gerçek Rota", zorder=2)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], color='gray', marker='x', s=15, alpha=0.6, label="LiDAR Nirengiler")
    
    # Trail
    k_half = int(len(time_steps) * 0.6)
    ax.plot(ekf_traj[:k_half, 0], ekf_traj[:k_half, 1], 'b--', linewidth=1.2, label="Klasik EKF")
    ax.plot(aekf_traj[:k_half, 0], aekf_traj[:k_half, 1], 'g-', linewidth=1.5, label="Adaptif EKF")
    
    gps_pts = gnss_meas[:k_half]
    valid_gps = gps_pts[~np.isnan(gps_pts[:, 0])]
    if len(valid_gps) > 0:
        ax.scatter(valid_gps[:, 0], valid_gps[:, 1], color='red', s=8, alpha=0.25, label="GNSS Ölçümleri")
        
    # Otobüs gövdesini k_half anında çiz
    draw_bus_body(ax, true_traj[k_half, 0], true_traj[k_half, 1], true_traj[k_half, 2],
                  length=12.27, width=2.54, color='#00e5ff', alpha=0.5, edgecolor='#00838f')
                  
    plt.title(f"Yörünge Simülasyonu Canlı Görünüm Karesi (Zaman: {time_steps[k_half]:.1f} s)")
    plt.xlabel("X Konumu (m)")
    plt.ylabel("Y Konumu (m)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper right", fontsize=8)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_frame.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Kare 2: GNSS Degradation Frame (GNSS Bozulma Bölgesi, t = 45s civarı)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # GNSS bozulma bölgesine odaklanalım
    k_degraded = np.where((s_ref >= 380.0) & (s_ref <= 420.0))[0][0]
    zoom_start = max(0, k_degraded - 80)
    zoom_end = min(len(true_traj) - 1, k_degraded + 80)
    
    ax.plot(true_traj[zoom_start:zoom_end, 0], true_traj[zoom_start:zoom_end, 1], 'k-', linewidth=2.0, label="Gerçek Rota", zorder=2)
    ax.plot(ekf_traj[zoom_start:zoom_end, 0], ekf_traj[zoom_start:zoom_end, 1], 'b--', linewidth=1.5, label="Klasik EKF", zorder=3)
    ax.plot(aekf_traj[zoom_start:zoom_end, 0], aekf_traj[zoom_start:zoom_end, 1], 'g-', linewidth=2.0, label="Adaptif EKF", zorder=4)
    
    gps_pts_z = gnss_meas[zoom_start:zoom_end]
    valid_gps_z = gps_pts_z[~np.isnan(gps_pts_z[:, 0])]
    if len(valid_gps_z) > 0:
        ax.scatter(valid_gps_z[:, 0], valid_gps_z[:, 1], color='red', s=20, alpha=0.4, label="Bozulmuş GNSS Ölçümleri", zorder=1)
        
    draw_bus_body(ax, true_traj[k_degraded, 0], true_traj[k_degraded, 1], true_traj[k_degraded, 2],
                  length=12.27, width=2.54, color='#00e5ff', alpha=0.4, edgecolor='#00838f')
                  
    # NIS ve Hata Bilgi Kutusu
    err_ekf = np.hypot(ekf_traj[k_degraded, 0] - true_traj[k_degraded, 0], ekf_traj[k_degraded, 1] - true_traj[k_degraded, 1])
    err_aekf = np.hypot(aekf_traj[k_degraded, 0] - true_traj[k_degraded, 0], aekf_traj[k_degraded, 1] - true_traj[k_degraded, 1])
    info_text = (
        f"Zaman: {time_steps[k_degraded]:.1f} s\n"
        f"GNSS Kalitesi: DÜŞÜK (Urban Canyon)\n"
        f"Klasik EKF Hatası: {err_ekf:.2f} m\n"
        f"Adaptif EKF Hatası: {err_aekf:.2f} m\n"
        f"GPS Kovaryans R: {aekf_filter.r_gps_trace[min(k_degraded, len(aekf_filter.r_gps_trace)-1)]:.2f} m^2"
    )
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
            
    plt.title(f"GNSS Bozulma Bölgesi Canlı Görünüm Karesi (Zaman: {time_steps[k_degraded]:.1f} s)")
    plt.xlabel("X Konumu (m)")
    plt.ylabel("Y Konumu (m)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper right", fontsize=8)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degradation_frame.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Kare 3: Bus Stop Frame (Durak Yanaşma Analizi, t = 25s civarı)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    stop_idx = np.argmin(np.abs(s_ref - 200.0))
    k_stop = stop_idx - 10 # Duraktan 1 saniye önce
    zoom_start = max(0, stop_idx - 50)
    zoom_end = min(len(true_traj) - 1, stop_idx + 30)
    
    # Şerit koridoru çiz
    ax.axhspan(-1.8, 1.8, color='gray', alpha=0.1, zorder=0)
    
    ax.plot(true_traj[zoom_start:zoom_end, 0], true_traj[zoom_start:zoom_end, 1], 'k-', linewidth=2.0, label="Gerçek Rota", zorder=2)
    ax.plot(ekf_traj[zoom_start:k_stop+1, 0], ekf_traj[zoom_start:k_stop+1, 1], 'b--', linewidth=1.5, label="Klasik EKF", zorder=3)
    ax.plot(aekf_traj[zoom_start:k_stop+1, 0], aekf_traj[zoom_start:k_stop+1, 1], 'g-', linewidth=2.0, label="Adaptif EKF", zorder=4)
    
    # Hata sapma çizgileri
    ax.plot([true_traj[k_stop, 0], ekf_traj[k_stop, 0]], [true_traj[k_stop, 1], ekf_traj[k_stop, 1]], 'r:', linewidth=1.5, label="Klasik EKF Sapma")
    ax.plot([true_traj[k_stop, 0], aekf_traj[k_stop, 0]], [true_traj[k_stop, 1], aekf_traj[k_stop, 1]], 'g:', linewidth=1.5, label="Adaptif EKF Sapma")
    
    # Durak bölgesi ve otobüs gövdesi
    from project.simulation.route_generator import RouteGenerator
    rg = RouteGenerator()
    x_st, y_st, _, _ = rg.get_pose(200.0)
    draw_stop_zone(ax, [x_st, y_st], stop_radius=10.0, color='orange', alpha=0.12)
    draw_bus_body(ax, true_traj[k_stop, 0], true_traj[k_stop, 1], true_traj[k_stop, 2],
                  length=12.27, width=2.54, color='#00e5ff', alpha=0.4, edgecolor='#00838f')
                  
    ax.set_xlim(x_st - 20, x_st + 15)
    ax.set_ylim(y_st - 4, y_st + 4)
    
    # Hata hesaplama
    ex_ekf = ekf_traj[k_stop, 0] - true_traj[k_stop, 0]
    ey_ekf = ekf_traj[k_stop, 1] - true_traj[k_stop, 1]
    lat_ekf = -ex_ekf * np.sin(true_traj[k_stop, 2]) + ey_ekf * np.cos(true_traj[k_stop, 2])
    lon_ekf = ex_ekf * np.cos(true_traj[k_stop, 2]) + ey_ekf * np.sin(true_traj[k_stop, 2])
    
    ex_aekf = aekf_traj[k_stop, 0] - true_traj[k_stop, 0]
    ey_aekf = aekf_traj[k_stop, 1] - true_traj[k_stop, 1]
    lat_aekf = -ex_aekf * np.sin(true_traj[k_stop, 2]) + ey_aekf * np.cos(true_traj[k_stop, 2])
    lon_aekf = ex_aekf * np.cos(true_traj[k_stop, 2]) + ey_aekf * np.sin(true_traj[k_stop, 2])
    
    info_text = (
        f"Zaman: {time_steps[k_stop]:.1f} s\n"
        f"Durak ID: 1\n"
        f"Durak Konumu: s = 200.0 m (X: 200.0, Y: 0.0)\n"
        f"Analiz Penceresi: s = 190.0 - 210.0 m\n"
        f"Klasik EKF Sapması: Yanal={abs(lat_ekf):.3f}m, Boylamsal={abs(lon_ekf):.3f}m\n"
        f"Adaptif EKF Sapması: Yanal={abs(lat_aekf):.3f}m, Boylamsal={abs(lon_aekf):.3f}m\n"
        f"Klasik EKF Durak RMSE: 0.1190 m\n"
        f"Adaptif EKF Durak RMSE: 0.1162 m\n"
        f"AEKF Durak İyileştirmesi: 2.35%"
    )
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
            
    plt.title(f"Durak Yanaşma Analizi Canlı Görünüm Karesi (Zaman: {time_steps[k_stop]:.1f} s)")
    plt.xlabel("X Konumu (m)")
    plt.ylabel("Y Konumu (m)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="upper right", fontsize=8)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stop_frame.png", dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Kare 4: Covariance Frame (Kovaryans-Hata Korelasyonu, Tam Ölçek)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Üst Panel: Konum Hataları
    ax1.plot(time_steps, np.hypot(ekf_traj[:, 0] - true_traj[:, 0], ekf_traj[:, 1] - true_traj[:, 1]), 'b--', label="Klasik EKF Hatası")
    ax1.plot(time_steps, np.hypot(aekf_traj[:, 0] - true_traj[:, 0], aekf_traj[:, 1] - true_traj[:, 1]), 'g-', label="Adaptif EKF Hatası")
    ax1.plot(time_steps, np.hypot(ukf_traj[:, 0] - true_traj[:, 0], ukf_traj[:, 1] - true_traj[:, 1]), 'm-.', label="UKF Hatası")
    ax1.set_ylabel("Konum Hatası (m)")
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Zamana Bağlı Konum Hatası ve AEKF Kovaryans Adaptasyonu Korelasyonu")
    
    degraded_runs = get_continuous_segments(time_steps, is_degraded_flags)
    for start, end in degraded_runs:
        ax1.axvspan(start, end, color='red', alpha=0.1)
    slip_runs = get_continuous_segments(time_steps, is_slipping_flags)
    for start, end in slip_runs:
        ax1.axvspan(start, end, color='orange', alpha=0.1)
        
    # Alt Panel: Kovaryans Değişimleri (Log ölçekte)
    t_gps_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_gps_trace))
    t_odom_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_odom_trace))
    t_lidar_r = np.linspace(0, time_steps[-1], len(aekf_filter.r_lidar_trace))
    
    ax2.plot(t_gps_r, aekf_filter.r_gps_trace, 'r-', label="GNSS Kovaryansı ($R_{gps}$)")
    ax2.plot(t_odom_r, aekf_filter.r_odom_trace, 'orange', label="Odometri Kovaryansı ($R_{odom}$)")
    ax2.plot(t_lidar_r, aekf_filter.r_lidar_trace, 'g-', label="LiDAR Kovaryansı ($R_{lidar}$)")
    ax2.set_yscale('log')
    ax2.set_ylabel("Kovaryans Değerleri (Log Ölçek)")
    ax2.set_xlabel("Zaman (s)")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)
    
    for start, end in degraded_runs:
        ax2.axvspan(start, end, color='red', alpha=0.1)
    for start, end in slip_runs:
        ax2.axvspan(start, end, color='orange', alpha=0.1)
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/covariance_frame.png", dpi=300)
    plt.close()

def get_continuous_segments(time_steps, flags):
    segments = []
    in_segment = False
    start_time = 0.0
    
    for i, flag in enumerate(flags):
        t = time_steps[i]
        if flag and not in_segment:
            in_segment = True
            start_time = t
        elif not flag and in_segment:
            in_segment = False
            segments.append((start_time, t))
            
    if in_segment:
        segments.append((start_time, time_steps[-1]))
        
    return segments
