import numpy as np


class ValueIteration:
    def __init__(self, discount_factor=0.9, goal_pos=(2, 2), pit_pos=(2, 1),wall_pos=(1, 1)):
        self.discount_factor = discount_factor
        self.value_table=np.zeros((3,3))
        self.value_table[goal_pos]=1
        self.value_table[pit_pos]=-1
        self.value_table[wall_pos]=0
        
    def print_value_table(self):
        print(np.round(self.value_table, 2))