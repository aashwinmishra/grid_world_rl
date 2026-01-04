import numpy as np


def q_learning(grid_world,
               initial_state: tuple,
               target_state: tuple,
               gamma: float,
               epsilon_start: float=1.0,
               epsilon_end: float= 0.1,
               epsilon_decay: float=0.9,
               alpha: float=0.1,
               num_episodes: int=50):
  """
  Executes Q Learning to find 'optimal' path between initial and target states.
  Args:
    grid_world: Instance of GridWorld class.
    initial_state: Starting position of the agent on the grid.
    target_state: Desired position of the agent on the grid.
    gamma: Discount factor.
    epsilon_start: Initial parameter for epsilon-greedy policy version.
    epsilon_end: Lowest value of epsilon.
    epsilon_decay: Ratio by which to decrease epsilon over every episode.
    alpha: 'learning rate' for RM algorithm.
    num_episodes: Number of episodes to use in policy/path determination.
  Returns:
    final policy learnt to chart path between initial_state and target_state. 
  """

  states = [state for state in grid_world.get_valid_states() if state not in grid_world.terminal]
  actions = grid_world.actions
  Q_table = np.zeros((len(actions), grid_world.height, grid_world.width))             #Initialization of Q table of shape [num_actions, H, W]
  epsilon = epsilon_start
  for episode in range(num_episodes):                                                 #Outer loop over N episodes
    current_state = initial_state                                                     #Position initialized for each episode
    epsilon = max(epsilon_end, epsilon_decay * epsilon)
    while current_state != target_state and current_state not in grid_world.terminal: #Inner loop for each episode
      action, reward, new_state = soft_policy_step(grid_world, 
                                                   current_state, 
                                                   Q_table, epsilon)                  #a, r, s' from epsilon greedy step.
      Q_table[actions.index(action), current_state[0], current_state[1]] -= \
      alpha * (Q_table[actions.index(action), current_state[0], current_state[1]] 
               - (reward + gamma * max(Q_table[:, new_state[0], new_state[1]])))      #RM step update for Q table
      current_state = new_state 
  policy = np.full((grid_world.height, grid_world.width), "", dtype='<U5')            #Random Initialization of policy
  for state in states:
    policy[state] = actions[np.argmax(Q_table[:, state[0], state[1]], axis=0)]        #Update policy.
  return policy


def soft_policy_step(grid_world, 
                     state: tuple, 
                     Q_table: np.array, 
                     epsilon: float):
  """
  Takes 1 step in grid world instance using an epsilon greedy policy.
  Args:
    grid_world: Instance of GridWorld class.
    state: Current position of agent.
    Q_table: Current table of state action values.
    epsilon: Parameter for epsilon-greedy policy version.
  Returns:
    tuple of selected action, reward and new state.
  """
  N = len(grid_world.actions)
  p = epsilon / N
  pstar = 1 - epsilon * (N - 1)/N
  probs = [p for _ in grid_world.actions]
  action = grid_world.actions[np.argmax(Q_table[:, state[0], state[1]], axis=0)]
  probs[grid_world.actions.index(action)] = pstar
  selected_action = np.random.choice(grid_world.actions, p=probs).item()
  new_state, reward = grid_world.sample(state, selected_action)
  return selected_action, reward, new_state


