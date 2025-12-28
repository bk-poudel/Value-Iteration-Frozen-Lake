import pygame
import sys
import numpy as np
import os
# --- DRIVER FIX ---
# Forces software rendering to bypass "libGL error: failed to load driver: iris"
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_RENDER_DRIVER'] = 'software'
class ValueIteration:
    def __init__(self, immediate_reward=-0.04, discount_factor=0.9,
                 goal_pos=(0, 2), pit_pos=(1, 2), wall_pos=(1, 1)):
        self.immediate_reward = immediate_reward
        self.discount_factor = discount_factor
        self.grid_size = (3, 3)
        self.goal_pos = goal_pos
        self.pit_pos = pit_pos
        self.wall_pos = wall_pos
        # Initialize values
        self.value_table = np.zeros(self.grid_size)
        self.value_table[goal_pos] = 1.0
        self.value_table[pit_pos] = -1.0
    def update_value_table_single_step(self):
        new_value_table = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for i in range(3):
            for j in range(3):
                # IMPORTANT: Skip Terminal States and Walls
                if (i, j) == self.goal_pos or (i, j) == self.pit_pos or (i, j) == self.wall_pos:
                    continue
                max_value = float("-inf")
                for action in actions:
                    next_state = (i + action[0], j + action[1])
                    # Boundary Check or Wall Collision
                    if not (0 <= next_state[0] < 3 and 0 <= next_state[1] < 3) or next_state == self.wall_pos:
                        next_state = (i, j)
                    # Bellman Equation: R + gamma * V(s')
                    v_next = self.value_table[next_state]
                    expected_value = self.immediate_reward + (self.discount_factor * v_next)
                    if expected_value > max_value:
                        max_value = expected_value
                new_value_table[i, j] = max_value
        self.value_table = new_value_table
    def get_best_action(self, i, j):
        if (i, j) in [self.goal_pos, self.pit_pos, self.wall_pos]:
            return None
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        action_names = ["↑", "↓", "←", "→"]
        best_val = float("-inf")
        best_act = None
        for idx, action in enumerate(actions):
            ni, nj = i + action[0], j + action[1]
            if not (0 <= ni < 3 and 0 <= nj < 3) or (ni, nj) == self.wall_pos:
                ni, nj = i, j
            val = self.immediate_reward + self.discount_factor * self.value_table[ni, nj]
            if val > best_val:
                best_val = val
                best_act = action_names[idx]
        return best_act
# ---------------------------------------------------------
# Pygame visualization
# ---------------------------------------------------------
CELL_SIZE = 150
WIDTH, HEIGHT = 3 * CELL_SIZE, 3 * CELL_SIZE
FPS = 10  # Slower FPS to see updates if animated
# Colors
WHITE = (255, 255, 255)
GRAY = (220, 220, 220)
BLACK = (0, 0, 0)
GOAL_COLOR = (100, 255, 100)
PIT_COLOR = (255, 100, 100)
WALL_COLOR = (50, 50, 50)
def main():
    pygame.init()
    # Try-except block for display to catch headless environment errors
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    except pygame.error:
        print("\n[!] Could not open a window. Using console output instead.\n")
        vi = ValueIteration()
        for k in range(10):
            vi.update_value_table_single_step()
            print(f"Iteration {k+1}:\n{vi.value_table}\n")
        return
    pygame.display.set_caption("Value Iteration - 3x3 Grid")
    font_large = pygame.font.SysFont("arial", 32, bold=True)
    font_small = pygame.font.SysFont("arial", 22)
    vi = ValueIteration()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Press SPACE to perform one iteration step
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                vi.update_value_table_single_step()
        screen.fill(WHITE)
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Determine Color
                color = GRAY
                if (i, j) == vi.goal_pos: color = GOAL_COLOR
                elif (i, j) == vi.pit_pos: color = PIT_COLOR
                elif (i, j) == vi.wall_pos: color = WALL_COLOR
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)
                # Draw Value
                val = vi.value_table[i, j]
                val_text = font_small.render(f"V: {val:.2f}", True, BLACK)
                screen.blit(val_text, (rect.x + 10, rect.y + 10))
                # Draw Best Action (Policy)
                action = vi.get_best_action(i, j)
                if action:
                    act_text = font_large.render(action, True, BLACK)
                    act_rect = act_text.get_rect(center=rect.center)
                    screen.blit(act_text, act_rect)
        pygame.display.flip()
        # Uncomment below to auto-run without pressing space
        vi.update_value_table_single_step()
        clock.tick(2) # Run at 2 steps per second
    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()