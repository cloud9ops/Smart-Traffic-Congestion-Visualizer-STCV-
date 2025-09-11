import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import networkx as nx

# ----- GNN model -----
class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        return x.mean(dim=0)  # graph-level embedding

# ----- Custom Gym environment -----
class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_nodes = 4
        self.edge_index = torch.tensor(
            [[0,0,0,1,1,2,2,3,3,3],
             [1,2,3,0,2,0,3,0,1,2]], dtype=torch.long)
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        # Observation = raw counts (4) + GNN embedding (8) = 12
        self.observation_space = gym.spaces.Box(
            low=-10, high=100, shape=(12,), dtype=np.float32)
        self.gnn = GNN(in_dim=1, hidden_dim=16, out_dim=8)

    def _build_graph(self):
        x = torch.tensor(self.state, dtype=torch.float).view(-1,1)
        return Data(x=x, edge_index=self.edge_index)

    def _get_obs(self):
        graph = self._build_graph()
        with torch.no_grad():
            emb = self.gnn(graph)
        return np.concatenate([self.state, emb.numpy()])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, 20, size=self.num_nodes)
        return self._get_obs(), {}

    def step(self, action):
        self.state[action] = max(self.state[action] - 5, 0)
        self.state += np.random.randint(0, 4, size=self.num_nodes)

        reward = -float(self.state.sum()) / self.num_nodes
        done = False
        return self._get_obs(), reward, done, False, {}

# ----- Train -----
env = TrafficEnv()
env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

model = DQN(
    "MlpPolicy", env,
    learning_rate=5e-4,
    buffer_size=10000,
    batch_size=64,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1
)
model.learn(total_timesteps=50000)

# ----- Test and visualize -----
def visualize_graph(state, step):
    G = nx.DiGraph()
    edges = [(0,1),(0,2),(0,3),(1,0),(1,2),(2,0),(2,3),(3,0),(3,1),(3,2)]
    G.add_edges_from(edges)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(4,4))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, labels={i: f"{i}\n{state[i]}" for i in range(4)})
    nx.draw_networkx_edges(G, pos, arrows=True)
    plt.title(f"Traffic Graph - Step {step}")
    plt.axis("off")
    plt.show()

obs, _ = env.reset()
for t in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, trunc, info = env.step(action)
    print(f"Step {t}, Action {action}, Reward {reward:.2f}")
    visualize_graph(env.env.state, t)  # <-- FIXED here

