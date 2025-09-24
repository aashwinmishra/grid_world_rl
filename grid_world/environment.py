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
    walls: List of tuples defining walls/forbidden states.
    slip_prob: probability of slipping perpendicular to the designated path.
  """
  def __init__(self, 
               size: tuple=(5,5),
               initial: tuple=(1,1),
               terminal: tuple=[(5,5)],
               rewards: list = [0],
               walls: list = [],
               step_cost: float=-0.02,
               slip_prob: float=0.05
               ):
    self.height, self.width = size 
    self.grid = np.zeros((self.height, self.width))
    self.actions = ["up", "down", "left", "right"]
    self.rewards = rewards 
    self.initial = initial 
    self.terminal = terminal
    self.walls = walls
    self.slip_prob = slip_prob
    self.step_cost = step_cost

  def step(self, 
           state: tuple, 
           action: str):
    numerical_action = {"up": (-1, 0),
                        "down": (1, 0),
                        "left": (0, -1),
                        "right": (0, 1)
                        }
    if state in self.terminal: #This part makes the move for nno-terminal states only.
      return state, 0
    new_state = state[0] + numerical_action[action][0], state[1] + numerical_action[action][1]
    if not (0 <= new_state[0] < self.height and 0 <= new_state[1] < self.width) or new_state in self.walls: #Moving beyond the grid 
      return state, self.step_cost
    return new_state, self.step_cost

  def get_transition_probs(self, state, action) -> list:
    answer = []
    stochastic_motions = {"up": (("up", 1 - 2 * self.slip_prob), ("right", self.slip_prob), ("left", self.slip_prob)),
                         "down": (("down", 1 - 2 * self.slip_prob), ("right", self.slip_prob), ("left", self.slip_prob)),
                         "right": (("right", 1 - 2 * self.slip_prob), ("up", self.slip_prob), ("down", self.slip_prob)),
                         "left": (("left", 1 - 2 * self.slip_prob), ("up", self.slip_prob), ("down", self.slip_prob)),
                        }
    transitions = stochastic_motions[action]
    for act, prob in transitions:
      new_state, reward = self.step(state, act)
      answer.append((prob, new_state, reward))
    return answer

  def get_valid_states(self):
    states = []
    for i in range(self.height):
      for j in range(self.width): 
        state = (i, j)
        if state not in self.walls:
          states.append(state)
    return states
