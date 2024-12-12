import torch as T
import numpy as np
from environment import Env
from nnAgent import DQNAgent
import dataloader
import matplotlib.pyplot as plt


data = dataloader.data
env = Env(data)


state_dim = 2 
n_actions = env.n_actions
agent = DQNAgent(state_dim=state_dim, action_dim=n_actions)


agent.q_network.load_state_dict(T.load("dqn_agent_model.pth", weights_only=True))
agent.q_network.eval()


test_episodes = 10 
positions = []
total_rewards = []
final_avts = []

for episode in range(test_episodes):
    state = env.reset()  
    total_reward = 0
    done = False
    episode_positions = []

    while not done:
        action = agent.select_action(state)  
        next_state, reward, done = env.step(action)  
        episode_positions.append(state)  
        state = next_state  
        total_reward += reward

    final_avt = env.new_location  
    final_avts.append(final_avt)
    positions.append(episode_positions)
    total_rewards.append(total_reward)
    print(f"Test Episode {episode + 1}/{test_episodes} - Total Reward: {total_reward:.2f} - Final AVT: {final_avt:.2f}")

# Sonuçları görselleştirme
for i, episode_positions in enumerate(positions):
    episode_positions = np.array(episode_positions)
    plt.plot(episode_positions[:, 0], episode_positions[:, 1], label=f'Episode {i + 1}')

plt.xlabel('d1 Position')
plt.ylabel('d2 Position')
plt.title('Agent Positions per Episode')
plt.legend()
plt.show()

# Ortalama ödül görselleştirme
plt.plot(range(1, test_episodes + 1), total_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

# Final AVT görselleştirme
plt.plot(range(1, test_episodes + 1), final_avts, marker='o')
plt.xlabel('Episode')
plt.ylabel('Final AVT')
plt.title('Final AVT per Episode')
plt.show()
