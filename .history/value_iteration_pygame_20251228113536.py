import pygame
import sys
import numpy as np
import os
# --- 1. THE DRIVER FIX ---
# This tells SDL (which Pygame uses) to avoid using the broken hardware GL driver
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_RENDER_DRIVER'] = 'software'
# In some remote cases, you might even need: os.environ['SDL_VIDEO_X11_VISUALID'] = '0x21'
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
                # Terminal states and walls do not get updated via Bellman
                if (i, j) == self.goal_pos or (i, j) == self.pit_pos or (i, j) == self.wall_pos:
                    continue
                max_value = float("-inf")
                for action in actions:
                    next_state = (i + action[0], j + action[1])
                    # Check boundaries or wall hit
                    if not (0 <= next_state[0] < 3 and 0 <= next_state[1] < 3) or next_state == self.wall_pos:
                        next_state = (i, j)
                    # Bellman: V(s) = R + gamma * V(s')
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
# --- Pygame Setup ---
CELL_SIZE = 140
WIDTH, HEIGHT = 3 * CELL_SIZE, 3 * CELL_SIZE
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200)
GOAL_COLOR, PIT_COLOR, WALL_COLOR = (100, 255, 100), (255, 100, 100), (50, 50, 50)
def main():
    vi = ValueIteration()
    # --- 2. THE FAILSAFE INITIALIZATION ---
    try:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Value Iteration Step-by-Step")
    except pygame.error as e:
        print(f"\n[!] Graphics Error: {e}")
        print("[!] Falling back to console-only output.\n")
        for step in range(10):
            vi.update_value_table_single_step()
            print(f"Iteration {step+1}:\n{vi.value_table}\n")
        return
    font = pygame.font.SysFont("arial", 24, bold=True)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Spacebar to step manually, or it will auto-run
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                vi.update_value_table_single_step()
        screen.fill(WHITE)
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = GRAY
                if (i, j) == vi.goal_pos: color = GOAL_COLOR
                elif (i, j) == vi.pit_pos: color = PIT_COLOR
                elif (i, j) == vi.wall_pos: color = WALL_COLOR
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)
                # Show Value Text
                val_text = font.render(f"{vi.value_table[i, j]:.2f}", True, BLACK)
                screen.blit(val_text, (rect.x + 10, rect.y + 10))
                # Show Action Arrow (The Policy)
                arrow = vi.get_best_action(i, j)
                if arrow:
                    a_text = font.render(arrow, True, BLACK)
                    screen.blit(a_text, a_text.get_rect(center=rect.center))
        pygame.display.flip()
        vi.update_value_table_single_step() # Auto-run iterations
        clock.tick(2) # 2 iterations per second
    pygame.quit()
if __name__ == "__main__":
    main()