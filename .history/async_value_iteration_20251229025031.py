import numpy as np
class AsynchronousValueIteration:
    def __init__(self, immediate_reward=-0.04, discount_factor=0.9, goal_pos=(0, 2), pit_pos=(1, 2), wall_pos=(1, 1)):
        self.immediate_reward = immediate_reward
        self.discount_factor = discount_factor
        # Initialize value table
        self.value_table = np.zeros((3, 3))
        # Set terminal and wall states
        self.goal_pos = goal_pos
        self.pit_pos = pit_pos
        self.wall_pos = wall_pos
        # Initial values for terminal states
        self.value_table[goal_pos] = 1.0
        self.value_table[pit_pos] = -1.0
        # The wall is technically not reachable, so it stays 0
    def print_value_table(self):
        # Rounding for readability
        print(np.round(self.value_table, 4))
    def bellman_equation(self, immediate_reward, next_state_value):
        return immediate_reward + self.discount_factor * next_state_value
    def update_value_table_asynchronous(self):
        """
        Updates the value table in-place. Changes made to one cell
        are immediately visible to the next cell in the same loop.
        """
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        max_delta = 0
        for i in range(3):
            for j in range(3):
                # Skip terminal states (Goal/Pit) and Walls
                if (i, j) == self.goal_pos or (i, j) == self.pit_pos or (i, j) == self.wall_pos:
                    continue
                old_v = self.value_table[i, j]
                max_value = float('-inf')
                for action in actions:
                    next_i, next_j = i + action[0], j + action[1]
                    # Boundary check and Wall check
                    if (next_i < 0 or next_i >= 3 or 
                        next_j < 0 or next_j >= 3 or 
                        (next_i, next_j) == self.wall_pos):
                        next_state = (i, j)
                    else:
                        next_state = (next_i, next_j)
                    # ASYNC STEP: Pull from the current, live self.value_table
                    next_state_value = self.value_table[next_state]
                    expected_value = self.bellman_equation(self.immediate_reward, next_state_value)
                    if expected_value > max_value:
                        max_value = expected_value
                # Update table immediately
                self.value_table[i, j] = max_value
                # Track the change for convergence check
                max_delta = max(max_delta, abs(old_v - self.value_table[i, j]))
        return max_delta
    def run_value_iteration(self, max_iterations=100, delta_threshold=0.0001):
        print("Starting Asynchronous Value Iteration...")
        print("Initial Table:")
        self.print_value_table()
        print("-" * 30)
        for iteration in range(1, max_iterations + 1):
            delta = self.update_value_table_asynchronous()
            print(f"Iteration {iteration} (Max Delta: {delta:.6f}):")
            self.print_value_table()
            if delta < delta_threshold:
                print(f"\nConverged in {iteration} iterations.")
                break
if __name__ == "__main__":
    # You will notice this converges faster than the synchronous version
    vi = AsynchronousValueIteration()
    vi.run_value_iteration()