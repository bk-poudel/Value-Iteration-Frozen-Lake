import os
# --- CRITICAL FIX FOR DRIVER ERROR ---
# This forces software rendering and avoids the broken OpenGL (GLX) path
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_RENDER_DRIVER'] = 'software'
# Prevents the crash on Opcode 149
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import pygame
import sys
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
        # Initialize values
        self.value_table = np.zeros(self.grid_size)
        self.value_table[goal_pos] = 1.0
        self.value_table[pit_pos] = -1.0
    def update(self):
        """Performs one iteration of the Bellman Optimality Equation."""
        new_v = np.copy(self.value_table)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for i in range(3):
            for j in range(3):
                # Skip terminal states (Goal/Pit) and Walls
                if (i, j) in [self.goal_pos, self.pit_pos, self.wall_pos]:
                    continue
                action_values = []
                for a in actions:
                    ni, nj = i + a[0], j + a[1]
                    # Boundary or Wall hit: stay in current square
                    if not (0 <= ni < 3 and 0 <= nj < 3) or (ni, nj) == self.wall_pos:
                        ni, nj = i, j
                    # Bellman: Reward + (Gamma * Next State Value)
                    v_next = self.value_table[ni, nj]
                    action_values.append(self.immediate_reward + self.discount_factor * v_next)
                new_v[i, j] = max(action_values)
        self.value_table = new_v
    def get_arrow(self, i, j):
        """Calculates which direction the agent should go based on values."""
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
# --- Visualization Settings ---
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
    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Value Iteration - 3x3 Grid")
    except pygame.error as e:
        print(f"\n[!] ERROR: Could not open window: {e}")
        print("Tip: If using SSH, ensure you used 'ssh -X' or 'ssh -Y'.")
        return
    font_v = pygame.font.SysFont("Arial", 24)
    font_a = pygame.font.SysFont("Arial", 50, bold=True)
    vi = ValueIteration()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Spacebar to step manually
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                vi.update()
        screen.fill(COLORS["bg"])
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(j * CELL, i * CELL, CELL, CELL)
                # Assign colors
                color = COLORS["bg"]
                if (i, j) == vi.goal_pos: color = COLORS["goal"]
                elif (i, j) == vi.pit_pos: color = COLORS["pit"]
                elif (i, j) == vi.wall_pos: color = COLORS["wall"]
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, COLORS["grid"], rect, 2)
                # Show Value
                val_text = font_v.render(f"V: {vi.value_table[i,j]:.2f}", True, COLORS["text"])
                screen.blit(val_text, (rect.x + 10, rect.y + 10))
                # Show Optimal Action Arrow
                arrow = vi.get_arrow(i, j)
                if arrow:
                    a_surf = font_a.render(arrow, True, COLORS["text"])
                    a_rect = a_surf.get_rect(center=rect.center)
                    screen.blit(a_surf, a_rect)
        pygame.display.flip()
        # Automatic step every 0.5 seconds
        vi.update()
        clock.tick(2) 
    pygame.quit()
if __name__ == "__main__":
    main()