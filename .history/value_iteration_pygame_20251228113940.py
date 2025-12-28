import os
import sys
# --- MANDATORY FIX FOR REMOTE/CLUSTER ENVIRONMENTS ---
# This forces software rendering and bypasses the hardware driver (GLX) crash.
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_RENDER_DRIVER'] = 'software'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import pygame
import numpy as np
class ValueIteration:
    def __init__(self, immediate_reward=-0.04, discount_factor=0.9,
                 goal_pos=(0, 2), pit_pos=(1, 2), wall_pos=(1, 1)):
        self.immediate_reward = immediate_reward
        self.discount_factor = discount_factor
        self.grid_size = (3, 3)
        self.goal_pos = goal_pos
        self.pit_pos = pit_pos
        self.wall_pos = wall_pos
        # Initialize Value Table
        self.value_table = np.zeros(self.grid_size)
        # Terminal states are constant
        self.value_table[goal_pos] = 1.0
        self.value_table[pit_pos] = -1.0
    def update(self):
        """Performs one sweep of the Bellman Optimality Equation."""
        new_v = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        for i in range(3):
            for j in range(3):
                # Do not update terminal states or the wall
                if (i, j) in [self.goal_pos, self.pit_pos, self.wall_pos]:
                    continue
                action_values = []
                for a in actions:
                    ni, nj = i + a[0], j + a[1]
                    # Boundary or Wall logic: stay in current square
                    if not (0 <= ni < 3 and 0 <= nj < 3) or (ni, nj) == self.wall_pos:
                        ni, nj = i, j
                    # Bellman: V(s) = R + gamma * V(s')
                    v_next = self.value_table[ni, nj]
                    action_values.append(self.immediate_reward + self.discount_factor * v_next)
                new_v[i, j] = max(action_values)
        self.value_table = new_v
    def get_policy(self, i, j):
        """Returns the best action arrow for the current value table."""
        if (i, j) in [self.goal_pos, self.pit_pos, self.wall_pos]: return ""
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        arrows = ["↑", "↓", "←", "→"]
        best_v = -float('inf')
        best_a = ""
        for idx, a in enumerate(actions):
            ni, nj = i + a[0], j + a[1]
            if not (0 <= ni < 3 and 0 <= nj < 3) or (ni, nj) == self.wall_pos:
                ni, nj = i, j
            if self.value_table[ni, nj] > best_v:
                best_v = self.value_table[ni, nj]
                best_a = arrows[idx]
        return best_a
# --- VISUALIZATION ---
CELL = 140
WIDTH, HEIGHT = 3 * CELL, 3 * CELL
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (220, 220, 220)
GOAL_COLOR, PIT_COLOR, WALL_COLOR = (100, 255, 100), (255, 100, 100), (60, 60, 60)
def main():
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Value Iteration - 3x3 Grid")
    except pygame.error as e:
        print(f"\n[!] ERROR: {e}")
        print("Tip: If you are using SSH, ensure you connected with 'ssh -X'.")
        return
    font = pygame.font.SysFont("Arial", 26, bold=True)
    vi = ValueIteration()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(WHITE)
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(j * CELL, i * CELL, CELL, CELL)
                # Assign colors
                color = GRAY
                if (i, j) == vi.goal_pos: color = GOAL_COLOR
                elif (i, j) == vi.pit_pos: color = PIT_COLOR
                elif (i, j) == vi.wall_pos: color = WALL_COLOR
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)
                # Show Value Text
                val_text = font.render(f"{vi.value_table[i,j]:.2f}", True, BLACK)
                screen.blit(val_text, (rect.x + 10, rect.y + 10))
                # Show Optimal Action Arrow
                arrow = vi.get_policy(i, j)
                if arrow:
                    a_surf = font.render(arrow, True, BLACK)
                    screen.blit(a_surf, a_surf.get_rect(center=rect.center))
        pygame.display.flip()
        vi.update()  # Continuous update
        clock.tick(2) # 2 iterations per second
    pygame.quit()
if __name__ == "__main__":
    main()