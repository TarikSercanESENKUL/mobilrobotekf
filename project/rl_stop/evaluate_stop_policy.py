import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Proje yollarını ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.rl_stop.stop_env import StopApproachEnv
from project.rl_stop.rl_utils import QNetwork

class RuleBasedStopController:
    """
    15 saniyelik durakta bekleme durum makinesine sahip kural tabanlı durak yaklaşma kontrolcüsü.
    """
    def __init__(self, dt=0.1, dwell_time=15.0):
        self.dt = dt
        self.dwell_time = dwell_time
        self.dwell_counter = 0.0
        self.stop_completed = False
        
    def reset(self):
        self.dwell_counter = 0.0
        self.stop_completed = False
        
    def get_action(self, d_stop, v):
        # Durak beklemesi bittiyse kalkış yap ve hızlan
        if self.stop_completed:
            return 3  # Hafif hızlanma (+0.5 m/s^2)
            
        # Duruş toleransına girildi ve hız sıfıra yakınsa dur/bekle
        if abs(d_stop) <= 0.5 and v <= 0.2:
            self.dwell_counter += self.dt
            if self.dwell_counter >= self.dwell_time:
                self.stop_completed = True
            return 2  # Sabit hız / dur (0.0 m/s^2)
            
        # Yaklaşma frenleme mantığı
        if d_stop > 30.0:
            return 2  # Sabit hız
        elif d_stop > 15.0:
            return 1  # Hafif fren (-1.0 m/s^2)
        else:
            return 0  # Güçlü fren (-2.5 m/s^2)

def run_single_episode_history(env, policy_type, policy_net=None, seed=42):
    state = env.reset(seed=seed)
    done = False
    
    d_history = [env.d_stop]
    v_history = [env.v]
    a_history = [env.a_prev]
    dwell_history = [env.dwell_counter]
    t_history = [0.0]
    
    rb_controller = RuleBasedStopController(dt=env.dt, dwell_time=env.dwell_time)
    
    t = 0.0
    while not done:
        if policy_type == "DQN":
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_idx = policy_net(state_t).argmax(1).item()
        else:
            action_idx = rb_controller.get_action(env.d_stop, env.v)
            
        state, reward, done, info = env.step(action_idx)
        
        t += env.dt
        d_history.append(env.d_stop)
        v_history.append(env.v)
        a_history.append(env.a_prev)
        dwell_history.append(env.dwell_counter)
        t_history.append(t)
        
    return t_history, d_history, v_history, a_history, dwell_history, info

def evaluate():
    print("Durak Yaklaşma ve Dwell Politikaları Değerlendirmesi Başlatılıyor...")
    
    env = StopApproachEnv(dt=0.1)
    
    # DQN modelini yükle
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "dqn_stop_model.pth")
    
    policy_net = QNetwork(env.state_size, env.action_size)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path))
        print("Eğitilmiş DQN modeli başarıyla yüklendi.")
    else:
        print("UYARI: Eğitilmiş model bulunamadı! train_dqn_stop.py çalıştırılmalı.")
    policy_net.eval()
    
    num_tests = 100
    methods = ["Rule-Based", "DQN"]
    results = {m: {} for m in methods}
    
    for method in methods:
        overshots = 0
        early_stops = 0
        successful_stops = 0
        dwell_completions = 0
        
        stop_errors = []
        dwell_times = []
        comfort_penalties = []
        
        rb_controller = RuleBasedStopController(dt=env.dt, dwell_time=env.dwell_time)
        
        for seed in range(2000, 2000 + num_tests):
            state = env.reset(seed=seed)
            rb_controller.reset()
            done = False
            
            a_prev = 0.0
            episode_jerk = []
            
            while not done:
                if method == "DQN":
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action_idx = policy_net(state_t).argmax(1).item()
                else:
                    action_idx = rb_controller.get_action(env.d_stop, env.v)
                    
                a_t = env.action_space[action_idx]
                episode_jerk.append(abs(a_t - a_prev))
                a_prev = a_t
                
                state, reward, done, info = env.step(action_idx)
                
            status = info.get("status", "unknown")
            final_d = env.d_stop
            
            stop_errors.append(abs(final_d))
            dwell_times.append(env.dwell_counter)
            comfort_penalties.append(np.mean(episode_jerk))
            
            if status == "overshot" or final_d < -1.0:
                overshots += 1
            elif status == "early_stop" or (final_d > 5.0 and env.dwell_counter < env.dwell_time):
                early_stops += 1
                
            # Eşik sınırları içinde sıfıra yakın hızla durma kontrolü
            if abs(final_d) <= env.stop_tolerance:
                successful_stops += 1
                
            if status == "success_dwell_complete" or env.dwell_counter >= env.dwell_time:
                dwell_completions += 1
                
        results[method] = {
            "method": method,
            "mean_stop_error_m": np.mean(stop_errors),
            "max_stop_error_m": np.max(stop_errors),
            "successful_stop_rate_percent": (successful_stops / num_tests) * 100.0,
            "dwell_completion_rate_percent": (dwell_completions / num_tests) * 100.0,
            "mean_dwell_time_s": np.mean(dwell_times),
            "overshot_count": overshots,
            "early_stop_count": early_stops,
            "mean_comfort_penalty": np.mean(comfort_penalties)
        }
        
    # Sonuçları CSV olarak kaydet
    df = pd.DataFrame(results).T
    results_dir = "project/outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path1 = os.path.join(results_dir, "rl_stop_results.csv")
    csv_path2 = os.path.join(results_dir, "stop_dwell_results.csv")
    
    df.to_csv(csv_path1, index=False)
    df.to_csv(csv_path2, index=False)
    
    print(f"Sonuçlar kaydedildi: {csv_path1} ve {csv_path2}")
    print(df.to_string())
    
    # ----------------------------------------------------
    # Grafik A: rl_stop_distance_velocity.png
    # ----------------------------------------------------
    t_hist, d_hist, v_hist, a_hist, dwell_hist, _ = run_single_episode_history(env, "DQN", policy_net, seed=42)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    
    # Mesafe Paneli
    ax1.plot(t_hist, d_hist, 'b-', linewidth=2.0, label="Durağa Kalan Mesafe")
    ax1.axhline(0.0, color='r', linestyle='--', alpha=0.7, label="Durak Noktası")
    # Durak bölgesini (son 15m) gölgelendir
    ax1.fill_between(t_hist, -2, 40, where=(np.array(d_hist) <= 15.0), color='orange', alpha=0.15, label="Durak Yanaşma Bölgesi (d <= 15m)")
    ax1.set_ylabel("Mesafe (m)")
    ax1.set_title("DQN Ajanı ile Durak Yaklaşma ve Dwell Profili (Seed: 42)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)
    
    # Hız Paneli
    ax2.plot(t_hist, v_hist, 'g-', linewidth=2.0, label="Hız")
    ax2.set_ylabel("Hız (m/s)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)
    
    # İvme / Dwell Paneli
    ax3.step(t_hist, a_hist, 'm-', where='post', linewidth=1.5, label="Fren/İvme Aksiyonu")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t_hist, dwell_hist, 'c-.', linewidth=1.5, label="Dwell Sayacı")
    ax3_twin.set_ylabel("Bekleme Süresi (s)")
    ax3.set_ylabel("İvme (m/s²)")
    ax3.set_xlabel("Zaman (s)")
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # Legend birleştirme
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)
    
    plt.tight_layout()
    fig_path1 = "project/outputs/figures/rl_stop_distance_velocity.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Mesafe-hız-dwell grafiği kaydedildi: {fig_path1}")
    
    # ----------------------------------------------------
    # Grafik B: rl_stop_dwell_behavior.png
    # ----------------------------------------------------
    # Bu grafikte zaman ekseninde: durağa kalan mesafe, hız, bekleme sayacı ve durum bilgileri gösterilir.
    fig, (ax_d, ax_v, ax_dw) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    
    ax_d.plot(t_hist, d_hist, 'b-', linewidth=2.0, label="Mesafe")
    ax_d.axhline(0.0, color='red', linestyle='--', alpha=0.7)
    # Dwell durumunu zaman ekseninde gölgelendir
    dwell_active = (np.array(dwell_hist) > 0.0)
    ax_d.fill_between(t_hist, -2, 40, where=dwell_active, color='green', alpha=0.15, label="DWELL_AT_STOP Aktif")
    ax_d.set_ylabel("Mesafe (m)")
    ax_d.set_title("Pekiştirmeli Öğrenme Ajanı Durak Bekleme (Dwell) Davranışı")
    ax_d.grid(True, linestyle=':', alpha=0.5)
    ax_d.legend(loc="upper right", fontsize=8)
    
    ax_v.plot(t_hist, v_hist, 'g-', linewidth=2.0, label="Hız")
    ax_v.fill_between(t_hist, 0, 10, where=dwell_active, color='green', alpha=0.15)
    ax_v.set_ylabel("Hız (m/s)")
    ax_v.grid(True, linestyle=':', alpha=0.5)
    ax_v.legend(loc="upper right", fontsize=8)
    
    ax_dw.plot(t_hist, dwell_hist, 'c-', linewidth=2.0, label="Dwell Sayaç Değeri (s)")
    ax_dw.axhline(15.0, color='red', linestyle=':', label="Hedef Süre (15s)")
    ax_dw.set_ylabel("Süre (s)")
    ax_dw.set_xlabel("Zaman (s)")
    ax_dw.grid(True, linestyle=':', alpha=0.5)
    ax_dw.legend(loc="upper left", fontsize=8)
    
    plt.tight_layout()
    fig_path2 = "project/outputs/figures/rl_stop_dwell_behavior.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Dwell bekleme davranışı zaman serisi grafiği kaydedildi: {fig_path2}")
    
    # ----------------------------------------------------
    # Grafik C: rl_vs_rule_based_stop_error.png
    # ----------------------------------------------------
    labels_err = ['Ortalama Hata (m)', 'Maksimum Hata (m)', 'Dwell Tamamlama Oranı (0-1)']
    
    rule_vals = [
        results["Rule-Based"]["mean_stop_error_m"], 
        results["Rule-Based"]["max_stop_error_m"], 
        results["Rule-Based"]["dwell_completion_rate_percent"] / 100.0
    ]
    dqn_vals = [
        results["DQN"]["mean_stop_error_m"], 
        results["DQN"]["max_stop_error_m"], 
        results["DQN"]["dwell_completion_rate_percent"] / 100.0
    ]
    
    x = np.arange(len(labels_err))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, rule_vals, width, label='Kural Tabanlı', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, dqn_vals, width, label='DQN (Önerilen)', color='#2ca02c', edgecolor='black')
    
    ax.set_ylabel('Performans Ölçütü')
    ax.set_title('Durak Yaklaşma/Dwell RL vs Kural Tabanlı Kontrol Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_err)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    fig_path3 = "project/outputs/figures/rl_vs_rule_based_stop_error.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Hata ve Dwell tamamlama karşılaştırma bar grafiği kaydedildi: {fig_path3}")
    
if __name__ == "__main__":
    evaluate()
