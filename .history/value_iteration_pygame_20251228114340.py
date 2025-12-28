import gymnasium as gym
import numpy as np
import time
class FrozenLakeAgent:
    def __init__(self, env_name="FrozenLake-v1", is_slippery=True):
        # We use 'human' render_mode to see the Gym window
        self.env = gym.make(env_name, is_slippery=is_slippery, render_mode="human")
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.gamma = 0.99
        self.value_table = np.zeros(self.n_states)
    def compute_value_iteration(self, delta_e=1e-8):
        """Calculates the long-term Value of every tile."""
        while True:
            updated_value_table = np.copy(self.value_table)
            for s in range(self.n_states):
                # Standard Bellman Update: V(s) = max_a Σ P(s'|s,a) [R + γV(s')]
                q_values = [
                    sum(prob * (reward + self.gamma * updated_value_table[next_s])
                        for prob, next_s, reward, _ in self.env.P[s][a])
                    for a in range(self.n_actions)
                ]
                self.value_table[s] = max(q_values)
            if np.max(np.abs(updated_value_table - self.value_table)) < delta_e:
                break
    def get_best_action(self, state):
        """Selects the action that points to the highest expected value."""
        q_values = [
            sum(prob * (reward + self.gamma * self.value_table[next_s])
                for prob, next_s, reward, _ in self.env.P[state][a])
            for a in range(self.n_actions)
        ]
        return np.argmax(q_values)
    def play_game(self, episodes=5):
        """Uses the Gym renderer to show the agent acting on its learned values."""
        for i in range(episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            print(f"Starting Episode {i+1}")
            while not (terminated or truncated):
                # 1. Choose action based on our Value Table
                action = self.get_best_action(state)
                # 2. Step the environment
                state, reward, terminated, truncated, _ = self.env.step(action)
                # The 'human' renderer handles the drawing automatically
                time.sleep(0.1) 
            if reward == 1:
                print("Goal Reached! 🏆")
            else:
                print("Fell in a hole... ❄️")
            time.sleep(1)
        self.env.close()
if __name__ == "__main__":
    agent = FrozenLakeAgent(is_slippery=True)
    print("Learning the Value Table...")
    agent.compute_value_iteration()
    print("Demonstrating Optimal Policy...")
    agent.play_game()