def update_value_table_single_step(self):
        # REMOVED: new_value_table = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for i in range(3):
            for j in range(3):
                # Skip terminal states and obstacles
                if (i, j) in [(0, 2), (1, 2), (1, 1)]: 
                    continue
                max_value = float('-inf')
                for action in actions:
                    next_i, next_j = i + action[0], j + action[1]
                    # Boundary check
                    if next_i < 0 or next_i >= 3 or next_j < 0 or next_j >= 3:
                        next_state = (i, j)
                    else:
                        next_state = (next_i, next_j)
                    # CRITICAL CHANGE: 
                    # We pull from self.value_table, which might have been 
                    # updated by the previous iteration of i or j in this SAME loop.
                    next_state_value = self.value_table[next_state]
                    expected_value = self.bellman_equation(self.immediate_reward, next_state_value, 1.0)
                    if expected_value > max_value:
                        max_value = expected_value
                # UPDATE IN-PLACE:
                self.value_table[i, j] = max_value
        print(f"Updated Value Table (Asynchronous): ")
        self.print_value_table()
        print("--------------------------------------------------")