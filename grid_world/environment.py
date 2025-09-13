import numpy as np


class GridWorld:
  """
  Defines the Grid World example enviornment.
  Attributes:
    height: Number of rows in the grid.
    width: Number of columns in the grid.
    grid: Numpy array representing the grid.
    actions: List of strings representing moves.
    rewards: Array defining the reward at each state.
    terminal: List of tuples defining terminal states on the grid.
    slip_prob: probability of slipping perpendicular to the designated path.
  """
  def __init__(self, 
               size: tuple=(5,5),
               initial: tuple=(1,1),
               terminal: tuple=[(5,5)],
               rewards: list,
               slip_prob: float=0.05
               ):
    """
    Initializes an instance of Grid World.
    Parameters:
      size: Tuple of number of rows and columns in the grid.
      initial: tuple of initial state.
      terminal: List of tuples defining terminal states on the grid.
      rewards: Array defining the reward at each state.
      slip_prob: probability of slipping perpendicular to the designated path.
    """
    self.height, self.width = size 
    self.grid = np.zeros((self.height, self.width))
    self.actions = ["up", "down", "left", "right"]
    self.rewards = rewards 
    self.initial = initial 
    self.terminal = terminal
    self.slip_prob = slip_prob

  def step(self, 
           state: tuple, 
           action: str) -> tuple:
    numerical_action = {"up": (-1, 0),
                        "down": (1, 0),
                        "left": (0, -1),
                        "right": (0, 1)
                        }
    if state in self.terminal:
      return state, 0
    else:
      new_state = state[0] + numerical_action[action][0], state[1] + numerical_action[action][1]
    if 0 <= new_state[0] < self.height and 0 <= new_state[1] < self.width:
      return new_state, self.rewards[new_state].item()
    else:
      return state, 0

  def get_transition_probs(self, state, action) -> list:
    answer = []
    stochastic_motions = {"up": (("up", 1 - 2 * self.slip_prob), ("right", self.slip_prob), ("left", self.slip_prob)),
                         "down": (("down", 1 - 2 * self.slip_prob), ("right", self.slip_prob), ("left", self.slip_prob)),
                         "right": (("right", 1 - 2 * self.slip_prob), ("up", self.slip_prob), ("down", self.slip_prob)),
                         "left": (("left", 1 - 2 * self.slip_prob), ("up", self.slip_prob), ("down", self.slip_prob)),
                        }
    transitions = stochastic_motions[action]
    for transition in transitions:
      new_state, reward = self.step(state, transition[0])
      answer.append((transition[1], new_state, reward))
    return answer

