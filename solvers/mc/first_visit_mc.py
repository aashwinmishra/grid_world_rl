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


def generate_episode_epsilon(grid_world, 
                     state, 
                     action, 
                     policy, 
                     length: int=10,
                     epsilon: float=0.5)->list:
  episode = []
  N = len(grid_world.actions)
  p = epsilon / N
  pstar = 1 - epsilon * (N - 1)/N
  for i in range(length):
    if state in grid_world.terminal:
      break
    new_state, reward = grid_world.sample(state, action)
    episode.append((state, action, reward))
    state = new_state
    a_star = policy[state]
    probs = [p for _ in grid_world.actions]
    probs[grid_world.actions.index(a_star)] = pstar
    action = np.random.choice(grid_world.actions, p=probs)
  return episode, state


def FirstVisit_Q_tables(grid_world, 
                            policy, 
                            Q_table, 
                            num_visits, 
                            gamma: float=0.9,
                            length: int=100):
  states = [state for state in grid_world.get_valid_states() if state not in grid_world.terminal]
  actions = grid_world.actions
  s, a = states[np.random.choice(len(states))], actions[np.random.choice(len(actions))]
  episode, final_state = generate_episode(grid_world, s, a, policy, length) 
  state_action_pairs_reversed = list(reversed([(s, a) for s,a,r in episode]))
  rewards_reversed = list(reversed([r for s,a,r in episode]))

  g = grid_world.rewards[final_state].item() if final_state in grid_world.terminal else 0.0
  for i in range(len(episode)):
    g = rewards_reversed[i] + gamma * g 
    s, a = state_action_pairs_reversed[i]
    if (s, a) not in state_action_pairs_reversed[i+1:]: #First visit to (s, a) in episode
      idx = actions.index(a)
      num_visits[idx, s[0], s[1]] += 1
      Q_table[idx, s[0], s[1]] += (1/ num_visits[idx, s[0], s[1]]) * (g - Q_table[idx, s[0], s[1]])
  return Q_table, num_visits 


def FirstVisit_PolicyImprovement(grid_world, 
                                 gamma: float=0.9, 
                                 length: int=25, 
                                 max_iter: int=25):
  states = grid_world.get_valid_states()
  actions = grid_world.actions
  policy = np.full((grid_world.height, grid_world.width), "", dtype='<U5')
  for state in states:
    if state not in grid_world.terminal:
      policy[state] = actions[np.random.choice([0, 1, 2, 3])]
  new_policy = policy.copy()
  Q_table = np.zeros((len(actions), grid_world.height, grid_world.width))
  num_visits = np.zeros((len(actions), grid_world.height, grid_world.width)) #to offset div by 0.
  
  for iter in range(max_iter):
    Q_table, num_visits = FirstVisit_Q_tables(grid_world, policy, Q_table, num_visits, gamma, length)
    for state in states:
      if state not in grid_world.terminal:
        new_policy[state] = actions[np.argmax(Q_table[:, state[0], state[1]], axis=0)]
    if np.array_equal(policy, new_policy): 
      print(f"Converged after {iter + 1} passes...")
      break
    policy = new_policy.copy()
  return policy

