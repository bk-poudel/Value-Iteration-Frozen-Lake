import numpy as np
class ValueIteration:
    def __init__(self, immediate_reward=-0.04,discount_factor=0.9, goal_pos=(0, 2), pit_pos=(1, 2),wall_pos=(1, 1)):
        self.immediate_reward = immediate_reward
        self.discount_factor = discount_factor
        self.value_table=np.zeros((3,3))
        self.value_table[goal_pos]=1
        self.value_table[pit_pos]=-1
        self.value_table[wall_pos]=0
    def print_value_table(self):
        print(self.value_table)
    def bellman_equation(self, immediate_reward, next_state_value, prob):
        return prob * (immediate_reward + self.discount_factor * next_state_value)
    def update_value_table_single_step(self):
        new_value_table = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for i in range(3):
            for j in range(3):
                max_value = float('-inf')
                for action in actions:
                    next_state=(i + action[0], j + action[1])
                    if next_state[0] < 0 or next_state[0] >= 3 or next_state[1] < 0 or next_state[1] >= 3:
                        next_state = (i, j)  # stay in place if out of bounds
                        next_state_value = self.value_table[next_state]
                    else:
                        next_state_value = self.value_table[next_state]
                    expected_value = self.bellman_equation(self.immediate_reward, next_state_value,1.0)
                    if expected_value > max_value:
                        max_value = expected_value
                new_value_table[i, j] = max_value
        self.value_table = new_value_table
    def run_value_iteration(self, iterations=100, delta_e=0.0001):
        for _ in range(iterations):
            old_value_table = np.copy(self.value_table)
            self.update_value_table_single_step()
            if np.max(np.abs(self.value_table - old_value_table)) < delta_e:
                break
if __name__ == "__main__":
    vi = ValueIteration()
    vi.run_value_iteration()
    vi.print_value_table()
