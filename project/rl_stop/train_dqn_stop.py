import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Proje yollarını ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.rl_stop.stop_env import StopApproachEnv
from project.rl_stop.rl_utils import QNetwork, ReplayBuffer

def train():
    print("DQN Ajanı Durak Yaklaşma Eğitimi Başlatılıyor...")
    
    # Ortamı başlat
    env = StopApproachEnv(dt=0.1)
    
    # Model hiperparametreleri
    episodes = 800
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.05
    batch_size = 64
    learning_rate = 1e-3
    target_update_freq = 5
    
    memory = ReplayBuffer(capacity=10000)
    
    policy_net = QNetwork(env.state_size, env.action_size)
    target_net = QNetwork(env.state_size, env.action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    episode_rewards = []
    success_history = []
    
    for ep in range(1, episodes + 1):
        state = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        
        while not done:
            # Epsilon-greedy aksiyon seçimi
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(env.action_size)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_idx = policy_net(state_t).argmax(1).item()
                    
            next_state, reward, done, info = env.step(action_idx)
            memory.push(state, action_idx, reward, next_state, float(done))
            
            state = next_state
            total_reward += reward
            
            # Ağ güncellemesi
            if len(memory) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                
                # Mevcut Q değerleri
                q_values = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                
                # Hedef Q değerleri
                with torch.no_grad():
                    max_next_q = target_net(next_states_b).max(1)[0]
                    expected_q = rewards_b + gamma * max_next_q * (1.0 - dones_b)
                    
                loss = criterion(q_values, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Epsilon azaltma
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(total_reward)
        status = info.get("status", "unknown")
        success_history.append(1 if "success" in status else 0)
        
        # Hedef ağı güncelle
        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if ep % 50 == 0:
            avg_rew = np.mean(episode_rewards[-50:])
            success_rate = np.mean(success_history[-50:]) * 100.0
            print(f"Episode: {ep:3d} | Ortalama Ödül (Son 50): {avg_rew:7.2f} | Başarı Oranı: %{success_rate:5.1f} | Epsilon: {epsilon:.3f}")
            
    # Eğitilen modeli kaydet
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "dqn_stop_model.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"DQN Ağı model ağırlıkları kaydedildi: {model_path}")
    
    # Grafik Çıktısı Hazırlama
    os.makedirs("project/outputs/figures/main", exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    # Hareketli ortalama alarak pürüzsüzleştir
    window = 15
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(episode_rewards, color='blue', alpha=0.2, label="Ham Bölüm Ödülü")
    plt.plot(range(window-1, len(episode_rewards)), smoothed_rewards, color='blue', linewidth=2, label="15 Adımlık Hareketli Ortalama")
    
    plt.title("DQN Ajanı Durak Yaklaşma Eğitim Eğrisi")
    plt.xlabel("Bölüm (Episode)")
    plt.ylabel("Toplam Ödül")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    figure_path = "project/outputs/figures/rl_stop_training_reward.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Eğitim ödülü grafiği kaydedildi: {figure_path}")
    
if __name__ == "__main__":
    train()
