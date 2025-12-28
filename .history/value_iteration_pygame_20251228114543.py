import gymnasium as gym
import numpy as np
import time
import sys
class FrozenLakeAgent:
    def __init__(self, env_name="FrozenLake-v1", is_slippery=True):
        # We use 'human' render_mode to see the character move in a window
        try:
            self.env = gym.make(env_name, is_slippery=is_slippery, render_mode="human")
        except Exception as e:
            print(f"Error initializing environment: {e}")
            sys.exit(1)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.gamma = 0.9
        self.value_table = np.zeros(self.n_states)
    def compute_value_iteration(self, delta_e=1e-8):
        """
        Calculates the long-term Value of every tile using the 
        Bellman Optimality Equation.
        """
        # Accessing .unwrapped bypasses the TimeLimit wrapper to get P
        P = self.env.unwrapped.P
        iteration = 0
        while True:
            iteration += 1
            updated_value_table = np.copy(self.value_table)
            for s in range(self.n_states):
                q_values = []
                for a in range(self.n_actions):
                    q_val = 0
                    # Sum over all possible outcomes (prob, next_state, reward, done)
                    for prob, next_s, reward, _ in P[s][a]:
                        q_val += prob * (reward + self.gamma * updated_value_table[next_s])
                    q_values.append(q_val)
                # The value of a state is the maximum expected reward of the best action
                self.value_table[s] = max(q_values)
            # Check for convergence
            if np.max(np.abs(updated_value_table - self.value_table)) < delta_e:
                print(f"Value Iteration converged after {iteration} iterations.")
                break
    def get_best_action(self, state):
        """Looks ahead one step and picks the action maximizing Value."""
        P = self.env.unwrapped.P
        q_values = []
        for a in range(self.n_actions):
            q_val = sum(prob * (reward + self.gamma * self.value_table[next_s])
                        for prob, next_s, reward, _ in P[state][a])
            q_values.append(q_val)
        return np.argmax(q_values)
    def play_game(self, episodes=5):
        """Uses the learned value_table to play the game."""
        for i in range(episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            print(f"--- Episode {i+1} ---")
            while not (terminated or truncated):
                # Choose the optimal action according to our Value Table
                action = self.get_best_action(state)
                # Take the step
                state, reward, terminated, truncated, _ = self.env.step(action)
                # Small pause so the human eye can follow the movement
                time.sleep(0.2) 
            if reward == 1:
                print("Result: Success! (Reached the Goal)")
            else:
                print("Result: Failed (Fell in a Hole)")
            time.sleep(1)
        self.env.close()
if __name__ == "__main__":
    # Create the agent
    # Set is_slippery=False if you want to see a deterministic path first
    agent = FrozenLakeAgent(is_slippery=True)
    print("Step 1: Computing Value Table via Value Iteration...")
    agent.compute_value_iteration()
    print("\nStep 2: Demonstrating the Optimal Policy in the Environment...")
    agent.play_game()