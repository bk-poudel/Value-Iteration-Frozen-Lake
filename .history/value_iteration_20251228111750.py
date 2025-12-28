import numpy as np
class ValueIteration:
    def __init__(self, discount_factor=0.9, goal_pos=(0, 2), pit_pos=(1, 2),wall_pos=(1, 1)):
        self.discount_factor = discount_factor
        self.value_table=np.zeros((3,3))
        self.value_table[goal_pos]=1
        self.value_table[pit_pos]=-1
        self.value_table[wall_pos]=0
    def print_value_table(self):
        print(self.value_table)
    def bellman_equation(self, immediate_reward, next_state_value, prob):
        return prob * (immediate_reward + self.discount_factor * next_state_value)
    def update_value_table_single_step():
        new_value_table = np.copy(self.value_table)
        for i in range(3):
            for j in range(3):
                if (i, j) in [(0, 2), (1, 2), (1, 1)]:
                    continue  # Skip goal, pit, and wall positions
                action_values = []
                # Up
                if i > 0:
                    action_values.append(self.bellman_equation(0, self.value_table[i-1, j], 1.0))
                else:
                    action_values.append(self.bellman_equation(0, self.value_table[i, j], 1.0))
                # Down
                if i < 2:
                    action_values.append(self.bellman_equation(0, self.value_table[i+1, j], 1.0))
                else:
                    action_values.append(self.bellman_equation(0, self.value_table[i, j], 1.0))
                # Left
                if j > 0:
                    action_values.append(self.bellman_equation(0, self.value_table[i, j-1], 1.0))
                else:
                    action_values.append(self.bellman_equation(0, self.value_table[i, j], 1.0))
                # Right
                if j < 2:
                    action_values.append(self.bellman_equation(0, self.value_table[i, j+1], 1.0))
                else:
                    action_values.append(self.bellman_equation(0, self.value_table[i, j], 1.0))
                new_value_table[i, j] = max(action_values)
        self.value_table = new_value_table
