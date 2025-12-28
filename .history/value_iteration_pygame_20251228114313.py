import pygame
import sys
import numpy as np
class ValueIteration:
    def __init__(self, immediate_reward=-0.04, discount_factor=0.9,
                 goal_pos=(0, 2), pit_pos=(1, 2), wall_pos=(1, 1)):
        self.immediate_reward = immediate_reward
        self.discount_factor = discount_factor
        self.value_table = np.zeros((3, 3))
        self.value_table[goal_pos] = 1
        self.value_table[pit_pos] = -1
        self.value_table[wall_pos] = 0
    def bellman_equation(self, immediate_reward, next_state_value, prob):
        return prob * (immediate_reward + self.discount_factor * next_state_value)
    def update_value_table_single_step(self):
        new_value_table = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for i in range(3):
            for j in range(3):
                if (i, j) in [(0, 2), (1, 2), (1, 1)]:  # goal, pit, wall
                    continue
                max_value = float("-inf")
                for action in actions:
                    next_state = (i + action[0], j + action[1])
                    if not (0 <= next_state[0] < 3 and 0 <= next_state[1] < 3):
                        next_state = (i, j)
                    next_state_value = self.value_table[next_state]
                    expected_value = self.bellman_equation(self.immediate_reward, next_state_value, 1.0)
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
# ---------------------------------------------------------
# Pygame visualization
# ---------------------------------------------------------
CELL_SIZE = 120
WIDTH, HEIGHT = 3 * CELL_SIZE, 3 * CELL_SIZE
FPS = 30
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
GOAL_COLOR = (0, 200, 0)
PIT_COLOR = (200, 0, 0)
WALL_COLOR = (70, 70, 70)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 20)
def draw_grid(vi: ValueIteration):
    screen.fill(WHITE)
    for i in range(3):
        for j in range(3):
            x, y = j * CELL_SIZE, i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            cell_pos = (i, j)
            # colors for special cells
            if cell_pos == (0, 2):
                color = GOAL_COLOR
            elif cell_pos == (1, 2):
                color = PIT_COLOR
            elif cell_pos == (1, 1):
                color = WALL_COLOR
            else:
                color = GRAY
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
            # value text
            val = vi.value_table[i, j]
            text = font.render(f"{val:.2f}", True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
def main():
    vi = ValueIteration()
    vi.run_value_iteration()
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw_grid(vi)
        pygame.display.flip()
    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()
