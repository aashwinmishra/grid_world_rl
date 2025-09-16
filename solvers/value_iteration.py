import numpy as np
from grid_world.environment import GridWorld


def value_iteration(grid_world, 
                    gamma: float=0.9, 
                    theta: float=1e-10):
  """
  Performs Value Iteration for solving the Grid World problem.
  Parameters:
    grid_world: an instance of the GridWorld class defining the problem.
    gamma: discount factor for the returns.
    theta: minimum threshold to stop Value Iteration.
  Returns:
    Tuple (V, P) for the state values and the deterministi policy for the grid.
  """
  Vk = np.zeros((grid_world.height, grid_world.width))
  Q = np.zeros((grid_world.height, grid_world.width, len(grid_world.actions)))
  P = np.zeros((grid_world.height, grid_world.width))

  Vk1 = Vk.copy()
  for iter in range(1000):
    for i in range(grid_world.height):
      for j in range(grid_world.width):
        if (i, j) not in grid_world.terminal:
          for k in range(len(grid_world.actions)):
            possible_outcomes = grid_world.get_transition_probs((i, j), grid_world.actions[k])
            q_val = 0.0
            for prob, new_state, reward in possible_outcomes:
              q_val += prob * (reward + gamma * Vk[new_state])
            Q[i, j, k] = q_val
        P[i, j] = np.argmax(Q[i, j, :])
        Vk1[i, j] = np.max(Q[i, j, :])
    max_diff = np.max(np.abs(Vk - Vk1))
    Vk = Vk1.copy()
    if max_diff < theta:
      print(f"iteration: {iter}, Theta: {max_diff}")
      break
  for state in grid_world.terminal:
    Vk[state] = grid_world.rewards[state]
  return Vk, P


def value_iteration_deterministic(grid_world, 
                    gamma: float=0.9, 
                    theta: float=1e-8):
  Vk = np.zeros((grid_world.height, grid_world.width))
  Q = np.zeros((grid_world.height, grid_world.width, len(grid_world.actions)))
  P = np.zeros((grid_world.height, grid_world.width))

  Vk1 = Vk.copy()
  for iter in range(1000):
    for i in range(grid_world.height):
      for j in range(grid_world.width):
        if (i, j) not in grid_world.terminal:
          for k in range(len(grid_world.actions)):
            new_state, immediate_reward = grid_world.step((i, j), grid_world.actions[k])
            Q[i, j, k] = immediate_reward + gamma * Vk[new_state]
        P[i, j] = np.argmax(Q[i, j, :])
        Vk1[i, j] = np.max(Q[i, j, :])
    max_diff = np.max(np.abs(Vk - Vk1))
    Vk = Vk1.copy()
    if max_diff < theta:
      print(f"iteration: {iter}, Theta: {max_diff}")
      break
  for state in grid_world.terminal:
    Vk[state] = grid_world.rewards[state]
  return Vk, P


