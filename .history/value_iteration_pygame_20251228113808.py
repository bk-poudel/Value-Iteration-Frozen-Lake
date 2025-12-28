import pygame
import sys
import numpy as np
import os
# --- MANDATORY DRIVER FIX ---
# Forces Pygame to use the CPU for drawing (Software Rendering)
# This bypasses the "libGL error: failed to load driver: iris" crash.
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
        # Initialize Value Table
        self.value_table = np.zeros(self.grid_size)
        self.value_table[goal_pos] = 1.0
        self.value_table[pit_pos] = -1.0
    def update(self):
        """Performs one full sweep of the Bellman Optimality Equation."""
        new_v = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for i in range(3):
            for j in range(3):
                # Skip terminal states and the wall
                if (i, j) in [self.goal_pos, self.pit_pos, self.wall_pos]:
                    continue
                v_options = []
                for a in actions:
                    ni, nj = i + a[0], j + a[1]
                    # Boundary or Wall logic
                    if not (0 <= ni < 3 and 0 <= nj < 3) or (ni, nj) == self.wall_pos:
                        ni, nj = i, j # Stay in place
                    v_options.append(self.immediate_reward + self.discount_factor * self.value_table[ni, nj])
                new_v[i, j] = max(v_options)
        self.value_table = new_v
    def get_policy(self, i, j):
        """Returns the best action arrow for a given cell."""
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
COLORS = {
    "bg": (255, 255, 255),
    "grid": (0, 0, 0),
    "goal": (100, 255, 100),
    "pit": (255, 100, 100),
    "wall": (60, 60, 60),
    "text": (0, 0, 0)
}
def main():
    vi = ValueIteration()
    try:
        pygame.init()
        # Attempting to open the window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Value Iteration Step-by-Step")
    except Exception as e:
        print(f"\n[!] Graphics Error: {e}")
        print("[!] Environment does not support windows. Outputting to console instead:\n")
        for k in range(10):
            vi.update()
            print(f"Iteration {k+1}:\n{vi.value_table}\n")
        return
    font_v = pygame.font.SysFont("Arial", 22)
    font_a = pygame.font.SysFont("Arial", 48, bold=True)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Spacebar to pause/step manually
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                vi.update()
        screen.fill(COLORS["bg"])
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(j*CELL, i*CELL, CELL, CELL)
                # Determine color
                color = COLORS["bg"]
                if (i, j) == vi.goal_pos: color = COLORS["goal"]
                elif (i, j) == vi.pit_pos: color = COLORS["pit"]
                elif (i, j) == vi.wall_pos: color = COLORS["wall"]
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, COLORS["grid"], rect, 2)
                # Render Value
                v_str = f"{vi.value_table[i,j]:.2f}"
                v_surf = font_v.render(v_str, True, COLORS["text"])
                screen.blit(v_surf, (j*CELL + 10, i*CELL + 10))
                # Render Policy Arrow
                arrow = vi.get_policy(i, j)
                if arrow:
                    a_surf = font_a.render(arrow, True, COLORS["text"])
                    a_rect = a_surf.get_rect(center=rect.center)
                    screen.blit(a_surf, a_rect)
        pygame.display.flip()
        vi.update()  # Continuous update
        clock.tick(2) # 2 steps per second
    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()