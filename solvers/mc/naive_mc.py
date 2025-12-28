import numpy as np


def generate_episode(grid_world, 
                     state, 
                     action, 
                     policy, 
                     length: int=10)->list:
  episode = []
  for i in range(length):
    if state in grid_world.terminal:
      break
    new_state, reward = grid_world.sample(state, action)
    episode.append((state, action, reward))
    state, action = new_state, policy[new_state]
  return episode, state


def get_q_value_estimate(grid_world, 
                         state, 
                         action, 
                         policy, 
                         gamma: float=0.9,
                         length: int=10, 
                         samples: int=5):
  q = 0.0 
  for _ in range(samples):
    episode, final_state = generate_episode(grid_world, state, action, policy, length)
    if len(episode) == 0:
      continue
    if final_state in grid_world.terminal:
      q_episode = grid_world.rewards[final_state].item()
    else:
      q_episode = 0.0
    rewards_only = [exp[2] for exp in episode]
    for reward in reversed(rewards_only):
      q_episode = reward + gamma * q_episode
    q += q_episode
  return q / samples 


def QValue_MC_Estimate(grid_world, 
                       policy, 
                       gamma: float=0.9,
                       length:int=10, 
                       samples: int=10):
  states = grid_world.get_valid_states()
  actions = grid_world.actions
  Q = np.zeros(( len(actions), grid_world.height, grid_world.width)) #[A, H, W]
  for state in states:
    if state not in grid_world.terminal:
      for a in range(len(actions)):
        Q[a, state[0], state[1]] = get_q_value_estimate(grid_world, state, actions[a], policy, gamma,length, samples)
  return Q


def MC_PolicyImprovement(grid_world, 
                         Q):
  states = grid_world.get_valid_states()
  actions = grid_world.actions
  policy = np.full((grid_world.height, grid_world.width), None)
  for state in states:
    if state not in grid_world.terminal:
      policy[state] = actions[np.argmax(Q[:, state[0], state[1]], axis=0)]
  return policy


def MC_naive_PolicyIteration(grid_world, 
                             gamma: float=0.9, 
                             length: int=25, 
                             samples:int=20, 
                             max_iter: int=20):
  states = grid_world.get_valid_states()
  actions = grid_world.actions
  policy = np.full((grid_world.height, grid_world.width), "", dtype='<U5')
  for state in states:
    if state not in grid_world.terminal:
      policy[state] = actions[np.random.choice([0, 1, 2, 3])]

  for iter in range(max_iter):
    Q = QValue_MC_Estimate(grid_world, policy, gamma, length, samples)
    new_policy = MC_PolicyImprovement(grid_world, Q)
    if np.array_equal(policy, new_policy): 
      break
    policy = new_policy
  return policy, Q
