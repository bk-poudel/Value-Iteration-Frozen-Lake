import numpy as np


class ValueIteration:
    def __init__(self, discount_factor=0.9):
        self.discount_factor = discount_factor
        self.value_table=np.zeros((3,3))