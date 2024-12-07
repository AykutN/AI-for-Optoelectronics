import torch as T
import numpy as np
from environment import Env
from nnAgent import DQNAgent
import dataloader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Hyperparametreler
EPISODES = 500  # Eğitim epizodu sayısı
MAX_STEPS = 250  # Her epizodda maksimum adım sayısı
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 250  # Kaç adımda bir target network güncellenecek

# Veriyi yükle ve çevreyi oluştur
data = dataloader.data
env = Env(data)

# Aracıyı oluştur
state_dim = 2  # d1 ve d2 pozisyonları
n_actions = env.n_actions
agent = DQNAgent(state_dim=state_dim, action_dim=n_actions, lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

# TensorBoard yazarını oluştur
writer = SummaryWriter()

# Eğitim döngüsü
episode_rewards = []  # Her epizodun toplam ödülünü saklayacak
episode_losses = []  # Her epizodun toplam loss değerini saklayacak
average_positions = []  # Her epizodun ortalama durulan konumunu saklayacak
final_positions = []  # Her epizodun bitirilen konumunu saklayacak

for episode in range(EPISODES):
    state = env.reset()  # Ortamı sıfırla
    total_reward = 0
    done = False
    episode_loss = 0  # Epizod başına toplam loss
    positions = []  # Epizod boyunca durulan konumlar

    for step in range(MAX_STEPS):
        action = agent.select_action(state)  # Aksiyon seç
        next_state, reward, done = env.step(action)  # Ortamı bir adım ilerlet

        # new_location değerini elde et
        new_location = env._get_location(env.d1p, env.d2p)
        highest_avt = env.highest_avt
        # Deneyimi belleğe ekle ve öğren
        agent.buffer.store(state, action, reward, next_state, done, new_location, highest_avt)
        loss = agent.train()
        
        if loss is not None:
            episode_loss += loss  # Loss değerini topla

        # Target ağı düzenli olarak güncelle
        if agent.step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        state = next_state  # Durumu güncelle
        total_reward += reward
        positions.append(state)  # Konumu sakla

        if done:
            break

    average_position = np.mean(positions, axis=0)
    average_positions.append(average_position)
    final_positions.append(state)
    episode_rewards.append(total_reward)
    episode_losses.append(episode_loss)

    # TensorBoard'a yaz
    writer.add_scalar('Total Reward', total_reward, episode)
    writer.add_scalar('Total Loss', episode_loss, episode)
    writer.add_scalar('Average Position X', average_position[0], episode)
    writer.add_scalar('Average Position Y', average_position[1], episode)

    print(f"Episode {episode + 1}/{EPISODES} - Total Reward: {total_reward:.2f} - Loss: {episode_loss:.4f} - Average Position: {average_position} - Final Position: {state} - Final AVT: {env._get_location(state[0], state[1])}")

# Modeli kaydet
T.save(agent.q_network.state_dict(), "dqn_agent_model.pth")
print("Model başarıyla kaydedildi!")

# TensorBoard yazarını kapat
writer.close()

# Eğitim sonuçlarını görselleştir
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance")
plt.legend()
plt.savefig("training_performance.png")  # Grafiği dosyaya kaydeder
plt.show()

# Loss değişimini görselleştir
plt.figure(figsize=(10, 5))
plt.plot(episode_losses, label="Total Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Loss")
plt.title("DQN Training Loss")
plt.legend()
plt.savefig("training_loss.png")  # Grafiği dosyaya kaydeder
plt.show()

# Ortalama durulan konumu ve bitirilen konumu yazdır
average_final_position = np.mean(final_positions, axis=0)
print(f"Average Final Position: {average_final_position}")

# Final konumların scatter plot'unu oluştur
final_positions_array = np.array(final_positions)

plt.figure(figsize=(10, 8))
plt.scatter(final_positions_array[:, 0], final_positions_array[:, 1], c='blue', marker='o', alpha=0.6)
plt.colorbar(label='Frequency')
plt.xlabel('d1 Position')
plt.ylabel('d2 Position')
plt.title('Scatter Plot of Final Positions')
plt.savefig("final_positions_scatter.png")  # Scatter plot'u dosyaya kaydeder
plt.show()


# Final AVT'lerin histogramını oluştur
final_avt_positions = [env._get_location(pos[0], pos[1]) for pos in final_positions if isinstance(pos, np.ndarray) and len(pos) == 2]
final_avt_positions_array = np.array(final_avt_positions)

# 45 üstü AVT değerlerini filtrele
filtered_avt_positions = final_avt_positions_array[final_avt_positions_array > 45]

# Filtrelenmiş AVT değerlerinin histogramını oluştur
plt.figure(figsize=(10, 8))
plt.hist(filtered_avt_positions, bins=50, color='red', alpha=0.7)
plt.xlabel('AVT Value')
plt.ylabel('Frequency')
plt.title('Histogram of Final AVT Values (Filtered > 45)')
plt.savefig("filtered_final_avt_histogram.png")  # Histogramı dosyaya kaydeder
plt.show()

# Final konumların heatmap'ini oluştur
heatmap, xedges, yedges = np.histogram2d(final_positions_array[:, 0], final_positions_array[:, 1], bins=(50, 50))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure(figsize=(10, 8))
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar(label='Frequency')
plt.xlabel('d1 Position')
plt.ylabel('d2 Position')
plt.title('Heatmap of Final Positions')
plt.savefig("final_positions_heatmap.png")  # Heatmap'i dosyaya kaydeder
plt.show()

# Final AVT'lerin histogramını oluştur
final_avt_positions = [env._get_location(pos[0], pos[1]) for pos in final_positions if isinstance(pos, np.ndarray) and len(pos) == 2]
final_avt_positions_array = np.array(final_avt_positions)

plt.figure(figsize=(10, 8))
plt.hist(final_avt_positions_array, bins=50, color='green', alpha=0.7)
plt.xlabel('AVT Value')
plt.ylabel('Frequency')
plt.title('Histogram of Final AVT Values')
plt.savefig("final_avt_histogram.png")  # Histogramı dosyaya kaydeder
plt.show()