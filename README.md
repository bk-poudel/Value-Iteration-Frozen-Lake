# Value Iteration on Frozen Lake
This repo demonstrates value iteration on the classic Frozen Lake problem from **Gymnasium** 
- `value_iteration.py` — runs value iteration on a small custom map without rendering or any GUI.
- `value_iteration_pygame.py` — identical core logic but exposes the `is_slippery` flag so you can switch between deterministic and stochastic transitions before watching the learned policy act.
Both scripts load the learned value function and then roll out a single episode using the greedy policy derived from the value estimates.
## Requirements
- Python 3.9+ (Gymnasium requires 3.9 or later)
- `gymnasium[toy-text]` for the Frozen Lake environment and textual renderer
- `numpy`
- A display is required if you run with `render_mode="human"`
```
python -m pip install gymnasium[toy-text] numpy
```
> 💡 Gymnasium uses `pygame` under the hood for human rendering. If it is missing, run `python -m pip install pygame`.
## Getting Started
1. (Optional) Create and activate a virtual environment.
2. Install the dependencies listed above.
3. Pick the script you want to experiment with and execute it:
```
python value_iteration.py
```
or
```
python value_iteration_pygame.py
```
## Customising the Map
Frozen-Lake expose a `MY_MAP` constant near the top of the file. You can replace it with any rectangular grid made from the following characters:
- `S` — start state (exactly one per map)
- `F` — frozen tiles the agent can safely traverse
- `H` — holes that end the episode with zero reward
- `G` — the goal state yielding reward 1
After editing the map, rerun the script to see how the optimal policy adapts. For larger maps you may want to relax the convergence tolerance or decrease `gamma` to speed things up.
## Tuning Value Iteration
Key hyperparameters live inside the `CustomFrozenLake` class:
- `gamma` — discount factor; keep it below 1 for convergence, lower for shorter planning horizons.
- Convergence threshold — set by the `1e-8` tolerance inside `compute_value_iteration`; increasing it accelerates convergence at the cost of accuracy.
Adjust these constants directly in the scripts to experiment with different planning behaviours.
## Troubleshooting
- **No render window appears**: ensure you are running on a machine with display access (or use an X server if remote) and that `pygame` installed successfully.
- **Import errors**: double-check the virtual environment and reinstall dependencies with `pip`.
- **Slow convergence**: increase the tolerance or reduce the map size while prototyping.
