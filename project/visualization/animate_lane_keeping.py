"""
Lane Keeping Analysis Animation
================================
Şerit takip analizi animasyonu — yanal hata zaman serisi ve coğrafi iz.

Çıktı: bus_lane_keeping.gif
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from project.simulation.route_generator import RouteGenerator
from project.simulation.lane_model import LaneModel
from project.visualization.animation_utils import (
    draw_route_corridor, draw_stop_zone, save_animation
)


def run_lane_simulation(total_time=180.0, dt=0.1):
    """
    Rota simülasyonu ve şerit metrikleri hesapla.
    """
    rg = RouteGenerator(dt=dt)
    time_steps, x_ref, y_ref, theta_ref, v_ref, s_ref, *extra = rg.generate_trajectory(total_time)
    
    route_waypoints = np.column_stack((x_ref, y_ref))
    lane_model = LaneModel(route_waypoints)
    
    N = len(time_steps)
    lateral_errors = []
    lane_departures = []
    
    rng = np.random.default_rng(42)
    
    # Adaptif EKF lokalizasyon gürültüsü simüle et (gerçekçi hata profili)
    for k in range(N):
        # EKF tahmini gürültüsü (konum bazlı)
        noise_x = rng.normal(0, 0.15)
        noise_y = rng.normal(0, 0.15)
        state = (x_ref[k] + noise_x, y_ref[k] + noise_y, theta_ref[k], v_ref[k])
        
        # Bazı bölgelerde daha fazla gürültü (GNSS bozulma)
        s_mod = s_ref[k] % rg.lap_len
        if 300 <= s_mod <= 500:
            extra_noise = rng.normal(0, 0.5)
            state = (state[0] + extra_noise, state[1] + extra_noise, state[2], state[3])
        
        lane_model.record_step(state, time_steps[k])
        lateral_errors.append(lane_model.lateral_errors[-1])
        lane_departures.append(lane_model.lane_departures[-1])
    
    metrics = lane_model.get_metrics()
    
    return {
        't': time_steps,
        'x': x_ref, 'y': y_ref, 'theta': theta_ref,
        'lateral_errors': lateral_errors,
        'lane_departures': lane_departures,
        'metrics': metrics,
        'route_waypoints': route_waypoints,
        'rg': rg,
    }


def make_lane_keeping_animation(frame_skip=8, output_path=None):
    """Şerit takip animasyonunu oluştur."""
    
    print("Şerit takip simülasyonu çalıştırılıyor...")
    data = run_lane_simulation()
    T = len(data['t'])
    frame_list = list(range(0, T, frame_skip))
    
    metrics = data['metrics']
    rg = data['rg']
    route_waypoints = data['route_waypoints']
    
    # Durak konumları
    stop_positions = {}
    for stop_s in rg.bus_stops:
        x_st, y_st, _, _ = rg.get_pose(stop_s)
        stop_positions[stop_s] = (x_st, y_st)
    
    print(f"Şerit metrikleri: {metrics}")
    
    # ─── Figür ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0f',
                              gridspec_kw={'width_ratios': [1.2, 1]})
    
    fig.suptitle('Şerit Takip Analizi — Adaptif EKF Lokalizasyonu',
                  color='#e0e0e0', fontsize=13, fontweight='bold', y=0.97)
    
    # ─── Sol panel: Harita ───────────────────────────────────────────────
    ax_map = axes[0]
    ax_map.set_facecolor('#0d1117')
    ax_map.set_aspect('equal')
    
    draw_route_corridor(ax_map, route_waypoints, corridor_width=7.2)
    
    for stop_s, (sx, sy) in stop_positions.items():
        draw_stop_zone(ax_map, (sx, sy), stop_radius=8.0, alpha=0.1)
    
    x_min, x_max = route_waypoints[:, 0].min() - 25, route_waypoints[:, 0].max() + 25
    y_min, y_max = route_waypoints[:, 1].min() - 25, route_waypoints[:, 1].max() + 25
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    ax_map.set_title('Araç Yörüngesi + Şerit', color='#aaaaaa', fontsize=10)
    ax_map.set_xlabel('X (m)', color='#777777', fontsize=9)
    ax_map.set_ylabel('Y (m)', color='#777777', fontsize=9)
    ax_map.tick_params(colors='#555555', labelsize=8)
    for sp in ax_map.spines.values():
        sp.set_color('#222222')
    
    # İz çizgisi
    trail_line, = ax_map.plot([], [], color='#00e5ff', linewidth=1.2, alpha=0.5, zorder=3)
    
    # Anlık pozisyon
    pos_scatter = ax_map.scatter([], [], s=60, c='#ff5252', marker='o',
                                  edgecolors='#ff8a80', linewidth=1.2, zorder=6)
    
    # Yanal hata okları (renk kodlu)
    lateral_arrow = [None]
    
    trail_x, trail_y = [], []
    
    # ─── Sağ panel: Hata zaman serisi ───────────────────────────────────
    ax_err = axes[1]
    ax_err.set_facecolor('#0d1117')
    
    # Şerit sınır çizgileri
    lane_half = 1.8
    ax_err.axhline(y=lane_half, color='#f57f17', linewidth=1.2, linestyle='--', alpha=0.6,
                    label='Şerit Sınırı (+1.8m)')
    ax_err.axhline(y=-lane_half, color='#f57f17', linewidth=1.2, linestyle='--', alpha=0.6,
                    label='Şerit Sınırı (-1.8m)')
    ax_err.axhline(y=0, color='#ffffff', linewidth=0.8, linestyle='-', alpha=0.2)
    
    # Şerit ihlal bölgesi rengi
    ax_err.fill_between(data['t'], lane_half, 3.0, alpha=0.06, color='#f44336')
    ax_err.fill_between(data['t'], -3.0, -lane_half, alpha=0.06, color='#f44336')
    
    # Hata çizgisi (tamamı düşük alpha ile)
    err_bg_line, = ax_err.plot(data['t'], data['lateral_errors'],
                                color='#00e5ff', linewidth=1.0, alpha=0.15)
    
    # Canlı hata çizgisi
    err_line, = ax_err.plot([], [], color='#00e5ff', linewidth=1.5, alpha=0.9)
    
    # Anlık hata nokta
    err_point = ax_err.scatter([], [], s=70, c='#ff5252', zorder=6)
    
    ax_err.set_xlim(0, data['t'][-1])
    ax_err.set_ylim(-3.0, 3.0)
    ax_err.set_title('Yanal Hata Zaman Serisi', color='#aaaaaa', fontsize=10)
    ax_err.set_xlabel('Zaman (s)', color='#777777', fontsize=9)
    ax_err.set_ylabel('Yanal Hata (m)', color='#777777', fontsize=9)
    ax_err.tick_params(colors='#555555', labelsize=8)
    for sp in ax_err.spines.values():
        sp.set_color('#222222')
    ax_err.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333355',
                   labelcolor='#cccccc', fontsize=7.5)
    
    # Metrik metin kutusu
    metric_text = ax_err.text(0.02, 0.95,
        f"Ort. Hata: {metrics.get('mean_lane_center_error_m', 0):.3f} m\n"
        f"Maks. Hata: {metrics.get('max_lane_center_error_m', 0):.3f} m\n"
        f"Şerit İhlali: {metrics.get('lane_departure_count', 0)} kez\n"
        f"Başarı Oranı: {metrics.get('lane_keeping_success_rate_percent', 0):.1f}%",
        transform=ax_err.transAxes, va='top', ha='left',
        color='#cccccc', fontsize=8.5, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                   edgecolor='#333355', alpha=0.9))
    
    # Zaman etiketi
    time_text = ax_err.text(0.98, 0.95, '', transform=ax_err.transAxes,
                              va='top', ha='right', color='#00e5ff', fontsize=9,
                              fontfamily='monospace')
    
    def update(frame_idx):
        k = frame_idx
        
        bx = data['x'][k]
        by = data['y'][k]
        t = data['t'][k]
        lat_err = data['lateral_errors'][k]
        is_departure = data['lane_departures'][k]
        
        # İz
        trail_x.append(bx)
        trail_y.append(by)
        if len(trail_x) > 100:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)
        
        # Pozisyon rengi — ihlal varsa kırmızı
        color = '#f44336' if is_departure else '#00e5ff'
        pos_scatter.set_offsets([[bx, by]])
        pos_scatter.set_color(color)
        
        # Hata çizgisi
        err_line.set_data(data['t'][:k+1], data['lateral_errors'][:k+1])
        err_point.set_offsets([[t, lat_err]])
        err_color = '#f44336' if is_departure else '#00e5ff'
        err_point.set_color(err_color)
        
        # Zaman etiketi
        time_text.set_text(f't={t:.1f}s | e={lat_err:.2f}m')
        
        return trail_line, pos_scatter, err_line, err_point, time_text
    
    anim = FuncAnimation(fig, update, frames=frame_list,
                          interval=80, blit=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'outputs', 'animations')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'bus_lane_keeping.gif')
    
    print(f"Lane keeping animasyonu kaydediliyor: {output_path}")
    save_animation(anim, output_path, fps=12)
    plt.close(fig)
    
    return output_path, metrics


if __name__ == '__main__':
    artifact_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'animations'
    )
    os.makedirs(artifact_dir, exist_ok=True)
    output_path = os.path.join(artifact_dir, 'bus_lane_keeping.gif')
    path, metrics = make_lane_keeping_animation(frame_skip=8, output_path=output_path)
    print(f"\nAnimasyon kaydedildi: {path}")
    print(f"Şerit metrikleri: {metrics}")
