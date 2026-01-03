import numpy as np


def q_learning(grid_world,
               initial_state: tuple,
               target_state: tuple,
               gamma: float,
               epsilon: float,
               alpha: float,
               num_episodes: int):
  sstates = [state for state in grid_world.get_valid_states() if state not in grid_world.terminal]
  actions = grid_world.actions
  policy = np.full((grid_world.height, grid_world.width), "", dtype='<U5')
  for state in states:
    policy[state] = actions[np.random.choice([0, 1, 2, 3])]
  Q_table = np.zeros((len(actions), grid_world.height, grid_world.width)) #[num_actions, H, W]
  for episode in range(num_episodes):
    current_state = initial_state
    while current_state != target_state and current_state not in grid_world.terminal:
      current_action = policy[current_state]
      action, reward, new_state = soft_policy_step(grid_world, current_state, current_action, epsilon)
      Q_table[actions.index(action), current_state[0], current_state[1]] -= \
      alpha * (Q_table[actions.index(action), current_state[0], current_state[1]] 
               - (reward + gamma * max(Q_table[:, new_state[0], new_state[1]])))
      policy[current_state] = actions[np.argmax(Q_table[:, current_state[0], current_state[1]], axis=0)]
      current_state = new_state 
  return policy


def soft_policy_step(grid_world, 
                     state: tuple, 
                     action: str, 
                     epsilon: float)->tuple:
  N = len(grid_world.actions)
  p = epsilon / N
  pstar = 1 - epsilon * (N - 1)/N
  probs = [p for _ in grid_world.actions]
  probs[grid_world.actions.index(action)] = pstar
  selected_action = np.random.choice(grid_world.actions, p=probs)
  new_state, reward = grid_world.sample(state, selected_action)
  return selected_action, reward, new_state
