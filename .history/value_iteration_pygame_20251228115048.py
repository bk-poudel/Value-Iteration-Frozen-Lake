import gymnasium as gym
import numpy as np
import time
# 1. DEFINE YOUR CUSTOM MAP HERE
# You can make this 3x3, 4x4, 8x8, etc.
# Just ensure it's a square or rectangle.
MY_MAP = [
    "FFFF",
    "FHHG", # The Goal 'G' is now here
    "HFHF",
    "FFFF"
]
class CustomFrozenLake:
    def __init__(self, map_layout, is_slippery=False):
        self.env = gym.make("FrozenLake-v1", desc=map_layout, is_slippery=is_slippery, render_mode="human")
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.gamma = 0.9
        self.value_table = np.zeros(self.n_states)
    def compute_value_iteration(self):
        P = self.env.unwrapped.P
        while True:
            updated_v = np.copy(self.value_table)
            for s in range(self.n_states):
                # Bellman Equation: max over actions
                self.value_table[s] = max([
                    sum(prob * (reward + self.gamma * updated_v[next_s])
                        for prob, next_s, reward, _ in P[s][a])
                    for a in range(self.n_actions)
                ])
            if np.max(np.abs(updated_v - self.value_table)) < 1e-8:
                break
    def play(self):
        state, _ = self.env.reset()
        done = False
        while not done:
            # Find best action based on learned values
            P = self.env.unwrapped.P
            action = np.argmax([
                sum(prob * (reward + self.gamma * self.value_table[next_s])
                    for prob, next_s, reward, _ in P[state][a])
                for a in range(self.n_actions)
            ])
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            time.sleep(0.3)
        self.env.close()
if __name__ == "__main__":
    agent = CustomFrozenLake(MY_MAP,is_slippery=False)
    agent.compute_value_iteration()
    print("Value Table Learned for Custom Map!")
    agent.play()