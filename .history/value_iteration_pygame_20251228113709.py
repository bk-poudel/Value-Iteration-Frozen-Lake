import pygame
import sys
import numpy as np
import os
# --- CRITICAL FIX FOR YOUR ERROR ---
# Forces software rendering and bypasses the hardware driver (iris/swrast) crash.
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_RENDER_DRIVER'] = 'software'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
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
    def update(self):
        new_v = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        for i in range(3):
            for j in range(3):
                # Terminal states and walls stay fixed
                if (i, j) in [self.goal_pos, self.pit_pos, self.wall_pos]:
                    continue
                action_values = []
                for a in actions:
                    ni, nj = i + a[0], j + a[1]
                    # Boundary or Wall collision: stay in current square
                    if not (0 <= ni < 3 and 0 <= nj < 3) or (ni, nj) == self.wall_pos:
                        ni, nj = i, j
                    # Bellman: R + (gamma * V_next)
                    v_next = self.value_table[ni, nj]
                    action_values.append(self.immediate_reward + self.discount_factor * v_next)
                new_v[i, j] = max(action_values)
        self.value_table = new_v
    def get_arrow(self, i, j):
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
# --- Visualization ---
CELL = 140
WIDTH, HEIGHT = 3 * CELL, 3 * CELL
pygame.init()
def main():
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Value Iteration Step-by-Step")
    except pygame.error as e:
        print(f"FAILED TO OPEN WINDOW: {e}")
        print("Try running: export SDL_VIDEODRIVER=dummy if you are in a purely terminal environment.")
        return
    font = pygame.font.SysFont("Arial", 25, bold=True)
    vi = ValueIteration()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((255, 255, 255))
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(j*CELL, i*CELL, CELL, CELL)
                # Colors
                color = (220, 220, 220)
                if (i, j) == vi.goal_pos: color = (100, 255, 100)
                elif (i, j) == vi.pit_pos: color = (255, 100, 100)
                elif (i, j) == vi.wall_pos: color = (50, 50, 50)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                # Value and Arrow
                val_text = font.render(f"{vi.value_table[i,j]:.2f}", True, (0,0,0))
                screen.blit(val_text, (j*CELL + 10, i*CELL + 10))
                arrow = vi.get_arrow(i, j)
                if arrow:
                    a_surf = font.render(arrow, True, (0, 0, 0))
                    screen.blit(a_surf, a_surf.get_rect(center=rect.center))
        pygame.display.flip()
        vi.update() # Run the algorithm
        clock.tick(2) # 2 steps per second so you can watch it learn
    pygame.quit()
if __name__ == "__main__":
    main()