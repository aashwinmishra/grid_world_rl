import numpy as np
from utils import policy_evaluation, policy_improvement


def policy_iteration(grid_world: GridWorld,
                     gamma: float=0.9,
                     theta: float=1e-6,
                     max_iter: int=500):
  policy = np.full((grid_world.height, grid_world.width), 1, dtype=int)
  for iter in range(max_iter):
    print(f"Policy Iteration Step {iter}")
    V = policy_evaluation(grid_world, policy, gamma)
    new_policy = policy_improvement(grid_world, V, gamma)
    if np.array_equal(policy, new_policy):
      break
    policy = new_policy.copy()
  return V, policy


