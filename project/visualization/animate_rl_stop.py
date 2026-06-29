"""
RL Stop Approach Animation
===========================
Eğitilmiş DQN modelini yükleyerek durak yaklaşma ortamında (StopApproachEnv)
bir epizod çalıştırır ve otobüsün duruş performansını canlı olarak canlandırır.

Görselleştirilenler:
1. Yol görünümü (Otobüsün durağa yaklaşması, durma tolerans kutusu, EKF gürültü bandı).
2. Hız ve mesafe zaman serisi.
3. İvme (aksiyon) komutları ve durakta bekleme süreci (dwell time).

Çıktı: project/outputs/animations/rl_stop_approach_simulation.gif
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation

# Proje yollarını ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.rl_stop.stop_env import StopApproachEnv
from project.rl_stop.rl_utils import QNetwork
from project.visualization.animation_utils import save_animation

def run_rl_episode(seed=42):
    """Eğitilmiş model ile tek bir epizot koşturur ve geçmiş verileri kaydeder."""
    env = StopApproachEnv(dt=0.1)
    
    # Model yükle
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(model_dir, "rl_stop", "dqn_stop_model.pth")
    
    policy_net = QNetwork(env.state_size, env.action_size)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    policy_net.eval()
    
    # Epizodu çalıştır
    state = env.reset(seed=seed)
    done = False
    
    history = {
        't': [0.0],
        'd_stop': [env.d_stop],
        'v': [env.v],
        'a': [env.a_prev],
        'dwell': [env.dwell_counter],
        'ekf_err': [env.ekf_error],
        'action_idx': [2], # default hover/sabit
        'status': ['starting']
    }
    
    t = 0.0
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_idx = policy_net(state_t).argmax(1).item()
            
        state, reward, done, info = env.step(action_idx)
        t += env.dt
        
        history['t'].append(t)
        history['d_stop'].append(env.d_stop)
        history['v'].append(env.v)
        history['a'].append(env.a_prev)
        history['dwell'].append(env.dwell_counter)
        history['ekf_err'].append(env.ekf_error)
        history['action_idx'].append(action_idx)
        history['status'].append(info.get('status', 'running'))
        
    return history, env

def make_rl_stop_animation(output_path=None, seed=42):
    print("DQN stop simülasyon verisi toplanıyor...")
    history, env = run_rl_episode(seed=seed)
    
    t_arr = np.array(history['t'])
    d_arr = np.array(history['d_stop'])
    v_arr = np.array(history['v'])
    a_arr = np.array(history['a'])
    dw_arr = np.array(history['dwell'])
    err_arr = np.array(history['ekf_err'])
    
    N = len(t_arr)
    # Animasyon hızı için frame atlama
    frame_skip = 4
    frames = list(range(0, N, frame_skip))
    if frames[-1] != N - 1:
        frames.append(N - 1)
        
    # ─── Figür Yapısı ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8), facecolor='#0a0a0f')
    
    # 1. Yol şematiği (1D)
    ax_road = plt.subplot2grid((3, 2), (0, 0), colspan=2, facecolor='#0d1117')
    # 2. Hız ve Mesafe grafiği
    ax_plots = plt.subplot2grid((3, 2), (1, 0), rowspan=2, facecolor='#0d1117')
    # 3. İvme (Aksiyon) ve dwell bar
    ax_actions = plt.subplot2grid((3, 2), (1, 1), rowspan=2, facecolor='#0d1117')
    
    # Başlık
    fig.suptitle('Pekiştirmeli Öğrenme (DQN) Durak Yaklaşma ve Duruş Kontrolü', 
                 color='#e0e0e0', fontsize=14, fontweight='bold', y=0.97)
    
    # ─── 1. Panel: Yol Görselleştirme ───────────────────────────────
    ax_road.set_xlim(-5, 45)
    ax_road.set_ylim(-3, 3)
    ax_road.axis('off')
    
    # Asfalt çizimi
    ax_road.axhspan(-1.5, 1.5, color='#1f2937', zorder=0)
    # Yol çizgileri
    ax_road.axhline(1.4, color='#ffffff', linewidth=1.5, alpha=0.8)
    ax_road.axhline(-1.4, color='#ffffff', linewidth=1.5, alpha=0.8)
    ax_road.axhline(0.0, color='#ffcb2b', linewidth=1.0, linestyle='--', alpha=0.5)
    
    # Durak Noktası (x = 0 konumunda)
    # Durağa kalan mesafe d_stop ise otobüsün konumu = d_stop olur.
    # Yani durak x=0 noktasında, otobüs sağdan sola (x=35'ten x=0'a) yaklaşır.
    ax_road.plot([0, 0], [-1.8, 1.8], color='#ef4444', linewidth=3, zorder=3)
    ax_road.text(0, 2.1, 'DURAK ÇİZGİSİ', color='#ef4444', fontsize=9, fontweight='bold', ha='center',
                 path_effects=[pe.withStroke(linewidth=2, foreground='#0a0a0f')])
    
    # Tolerans Bölgesi [0, 0.5m]
    rect_tol = patches.Rectangle((-0.5, -1.2), 1.0, 2.4, linewidth=1, 
                                 edgecolor='#10b981', facecolor='#10b981', alpha=0.15, zorder=1)
    ax_road.add_patch(rect_tol)
    ax_road.text(0, -2.0, 'Duruş Toleransı\n(±0.5m)', color='#10b981', fontsize=8, ha='center')
    
    # Stop Zone (15m bölgesi)
    rect_sz = patches.Rectangle((0, -1.35), 15.0, 2.7, linewidth=1, linestyle=':',
                                edgecolor='#f59e0b', facecolor='none', alpha=0.3, zorder=1)
    ax_road.add_patch(rect_sz)
    ax_road.text(7.5, -2.0, 'Frenleme Bölgesi (15m)', color='#f59e0b', fontsize=8, ha='center')
    
    # Otobüs çizim nesneleri
    # Otobüs genişliği 1D'de dikdörtgen olarak çizilecek. Arka aks (x konumunda)
    bus_width = 3.5
    bus_height = 1.0
    bus_rect = patches.Rectangle((40, -bus_height/2), bus_width, bus_height, 
                                 linewidth=1.5, edgecolor='#00e5ff', facecolor='#00e5ff', alpha=0.6, zorder=5)
    ax_road.add_patch(bus_rect)
    
    # EKF belirsizlik bandı (otobüs etrafında gölge)
    ekf_shade = patches.Rectangle((40, -bus_height/2 - 0.3), bus_width, bus_height + 0.6,
                                  linewidth=0, facecolor='#ef4444', alpha=0.15, zorder=4)
    ax_road.add_patch(ekf_shade)
    
    # ─── 2. Panel: Grafik Çizimleri ────────────────────────────────
    ax_plots.set_title('Araç Hız ve Mesafe Zaman Serisi', color='#aaaaaa', fontsize=10)
    ax_plots.set_xlabel('Zaman (s)', color='#777777', fontsize=9)
    ax_plots.set_ylabel('Değerler', color='#777777', fontsize=9)
    ax_plots.tick_params(colors='#555555', labelsize=8)
    for sp in ax_plots.spines.values():
        sp.set_color('#222222')
    ax_plots.grid(True, linestyle=':', color='#222222')
    
    line_dist, = ax_plots.plot([], [], color='#f59e0b', linewidth=2.0, label='Mesafe (m)')
    line_vel, = ax_plots.plot([], [], color='#00e5ff', linewidth=2.0, label='Hız (m/s)')
    ax_plots.set_xlim(0, t_arr[-1])
    ax_plots.set_ylim(-2, 40)
    ax_plots.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333355', labelcolor='#cccccc', fontsize=8.5)
    
    # ─── 3. Panel: Aksiyonlar ve Dwell ─────────────────────────────
    ax_actions.set_title('DQN Aksiyon ve Bekleme (Dwell) Durumu', color='#aaaaaa', fontsize=10)
    ax_actions.set_xlabel('Zaman (s)', color='#777777', fontsize=9)
    ax_actions.set_ylabel('İvme İsteği (m/s²)', color='#777777', fontsize=9)
    ax_actions.tick_params(colors='#555555', labelsize=8)
    for sp in ax_actions.spines.values():
        sp.set_color('#222222')
    ax_actions.grid(True, linestyle=':', color='#222222')
    
    line_acc, = ax_actions.plot([], [], color='#ef4444', linewidth=1.8, label='İvme (Aksiyon)')
    ax_actions.set_xlim(0, t_arr[-1])
    ax_actions.set_ylim(-3.0, 1.0)
    
    # Dwell progress bar arka planı
    dwell_bg = patches.Rectangle((0, 0.5), t_arr[-1], 0.3, facecolor='#1f2937', edgecolor='#374151', linewidth=1, zorder=1)
    ax_actions.add_patch(dwell_bg)
    dwell_bar = patches.Rectangle((0, 0.5), 0.0, 0.3, facecolor='#10b981', zorder=2)
    ax_actions.add_patch(dwell_bar)
    dwell_text = ax_actions.text(t_arr[-1]/2, 0.65, 'Yolcu Alma/Dwell: %0', color='#ffffff', 
                                 fontsize=8.5, ha='center', va='center', fontweight='bold')
    
    # Bilgi Kutusu
    info_box = ax_road.text(20, 2.2, '', color='#ffffff', fontsize=9, fontfamily='monospace', ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor='#475569', alpha=0.9))
    
    # Canlı çizim izleri
    trace_t, trace_d, trace_v, trace_a = [], [], [], []
    
    def update(frame_idx):
        k = frame_idx
        t = t_arr[k]
        d = d_arr[k]
        v = v_arr[k]
        a = a_arr[k]
        dw = dw_arr[k]
        ekf_err = err_arr[k]
        status = history['status'][k]
        
        # Otobüs pozisyonunu güncelle (1D düzlemde x = d_stop)
        # Otobüsün boyutu bus_width, d_stop otobüsün ön tamponu/merkezi olsun
        bus_rect.set_x(d - bus_width/2)
        
        # EKF belirsizlik bandı genişliği EKF hatasına göre değişsin
        ekf_width = bus_width + 2 * ekf_err
        ekf_shade.set_x(d - ekf_width/2)
        ekf_shade.set_width(ekf_width)
        
        # Otobüsün rengi durumuna göre değişsin
        if status == 'success_dwell_complete':
            bus_rect.set_facecolor('#10b981')
            bus_rect.set_edgecolor('#10b981')
        elif dw > 0:
            bus_rect.set_facecolor('#3b82f6')
            bus_rect.set_edgecolor('#3b82f6')
        elif d <= 15.0:
            bus_rect.set_facecolor('#f59e0b')
            bus_rect.set_edgecolor('#f59e0b')
        else:
            bus_rect.set_facecolor('#00e5ff')
            bus_rect.set_edgecolor('#00e5ff')
            
        # Grafikleri güncelle
        trace_t.append(t)
        trace_d.append(d)
        trace_v.append(v)
        trace_a.append(a)
        
        line_dist.set_data(trace_t, trace_d)
        line_vel.set_data(trace_t, trace_v)
        line_acc.set_data(trace_t, trace_a)
        
        # Dwell bar güncelle
        ratio = min(1.0, dw / env.dwell_time)
        dwell_bar.set_width(ratio * t_arr[-1])
        dwell_text.set_text(f'Yolcu Biniş/Dwell: %{ratio*100:.0f} ({dw:.1f}s / {env.dwell_time}s)')
        
        # Bilgi Kutusu
        state_str = 'CRUISE'
        if dw > 0:
            state_str = 'DURAKTA BEKLEME (DWELL)'
        elif d <= 15.0:
            state_str = 'KONTROLLÜ YAVAŞLAMA / FREN'
            
        info_str = (
            f"Zaman: {t:.1f}s | Konum: {d:.2f}m | Hız: {v:.2f} m/s ({v*3.6:.1f} km/h)\n"
            f"Aksiyon (İvme): {a:.2f} m/s² | EKF Konum Hatası: {ekf_err:.3f}m | Durum: {state_str}"
        )
        info_box.set_text(info_str)
        
        return bus_rect, ekf_shade, line_dist, line_vel, line_acc, dwell_bar, dwell_text, info_box
        
    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'outputs', 'animations')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'rl_stop_approach_simulation.gif')
        
    print(f"Animasyon kaydediliyor: {output_path}")
    save_animation(anim, output_path, fps=12)
    plt.close(fig)
    print("RL Animasyon Kaydedildi!")
    return output_path

if __name__ == '__main__':
    make_rl_stop_animation()
