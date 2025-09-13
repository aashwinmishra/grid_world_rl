import numpy as np
from .environment import GridWorld


def show_values(grid_world, V):
  n_rows, n_cols = grid_world.height, grid_world.width
  for i in range(n_rows):
    row = "|"
    for j in range(n_cols):
      row += f"{V[i,j]:^7.3f}" +"|"
    print(row)


def show_policy(grid_world, policy):
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

