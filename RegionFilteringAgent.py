__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import RLConfig as config

class RegionFilterAgent():

  image = None
  observation = None
  action = None
  reward = None
  cumReward = 0
  memory = []
  t = 0
  
  def __init__(self, qnet, learner=None):
    self.controller = qnet
    self.learner = learner

  def integrateObservation(self, obs):
    self.observation = obs
    self.action = None
    self.reward = None
    self.obsQueue = []

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None

    if self.observation != "NoChange":
      self.t += 1
      Z = np.zeros( (self.observation['features'].shape[0], self.observation['features'].shape[1] + 2) )
      Z[:,0:-2] = self.observation['features']
      Z[:,-2] = self.observation['areas']
      Z[:,-1] = float(self.t)/config.MAX_STEPS_ALLOWED
      values = self.controller.getActionValues( Z )
      actions = np.argmax(values, 1)
      maxValues = np.max(values, 1)
      for i in range(actions.shape[0]):
        self.obsQueue.append( [self.observation['indexes'][i], actions[i], maxValues[i]] + Z[i,:].tolist() )
      self.obsQueue.sort(key=lambda x: x[2], reverse=True)
    if len(self.obsQueue) > 0:
      self.action = self.obsQueue.pop(0)
    else:
      self.action = 'terminate'
    return self.action

  def giveReward(self, r):
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    self.reward = r
    self.cumReward += sum(r)

    # Replay Memory Format: boxIndex action maxValue features[62] reward
    self.memory.append( self.action + [r] )

  def reset(self):
    self.observation = None
    self.action = None
    self.reward = None
    self.obsQueue = []
    self.memory = []
    self.controller.loadNetwork()
    self.t = 0

  def learn(self):
    if self.learner != None:
      self.learner.learn(self.memory, self.controller)

