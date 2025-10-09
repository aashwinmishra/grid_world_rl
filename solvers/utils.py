import numpy as np


def policy_evaluation(grid_world: GridWorld, 
                      policy: np.array,
                      gamma: float=0.9,
                      theta: float=1e-10,
                      max_iter: int=500):
  Vk = np.zeros((grid_world.height, grid_world.width)) #Old Value Matrix
  for state in grid_world.terminal:
    Vk[state] = grid_world.rewards[state]
  Vk1 = Vk.copy() #New Value Matrix
  states = grid_world.get_valid_states()
  for iter in range(max_iter):
    for state in states:
      if state not in grid_world.terminal:
        possible_outcomes = grid_world.get_transition_probs(state, grid_world.actions[policy[state]])
        q_val = 0.0
        for prob, new_state, reward in possible_outcomes:
            q_val += prob * (reward + gamma * Vk[new_state])
        Vk1[state] = q_val
    max_diff = np.max(np.abs(Vk - Vk1))
    Vk = Vk1.copy()
    if max_diff < theta:
      print(f"Policy Evaluation converged at sub-iteration {iter} with Theta: {max_diff}")
      break
  return Vk


def policy_improvement(grid_world: GridWorld,
                       V: np.array,
                       gamma: float=0.9):
  policy = np.full((grid_world.height, grid_world.width), -1, dtype=int) #Current Policy
  states = grid_world.get_valid_states()
  for state in states:
    if state not in grid_world.terminal:
      Q = []
      for k in range(len(grid_world.actions)):
        possible_outcomes = grid_world.get_transition_probs(state, grid_world.actions[k])
        q_val = 0.0
        for prob, new_state, reward in possible_outcomes:
          q_val += prob * (reward + gamma * V[new_state])
        Q.append(q_val)
      policy[state] = np.argmax(Q)
  return policy
