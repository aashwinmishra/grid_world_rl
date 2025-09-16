import numpy as np
from .environment import GridWorld


def show_values(grid_world, V):
  """
  Renders a basic visualization of the state values for every position.
  Parameters:
    grid_world: An instance of the GridWorld class.
    V: Corresponding state values.
  Returns:
    None
  """
  n_rows, n_cols = grid_world.height, grid_world.width
  for i in range(n_rows):
    row = "|"
    for j in range(n_cols):
      row += f"{V[i,j]:^7.3f}" +"|"
    print(row)


def show_policy(grid_world, policy):
  """
  Renders a basic visualization of the given policy for every position.
  Parameters:
    grid_world: An instance of the GridWorld class.
    policy: Corresponding deterministic policy.
  Returns:
    None
  """
  action_symbols = {"up": "\u2191",
                    "down": "\u2193",
                    "right": "\u2192",
                    "left": "\u2190",
                    "stay": "o"
                    }
  n_rows, n_cols = grid_world.height, grid_world.width
  for i in range(n_rows):
    row = "|"
    for j in range(n_cols):
      if (i, j) not in grid_world.terminal:
        row += f"   {action_symbols[policy[i,j]]}   " +"|"
      else: 
        row += f"   {action_symbols["stay"]}   " +"|"
    print(row)

