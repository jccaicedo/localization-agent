__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.agents.logging import LoggingAgent
import numpy as np

#class ObjectLocalizationAgent(LoggingAgent):
class ObjectLocalizationAgent():

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
    self.image = obs[0]
    self.observation = obs[1]
    self.action = None
    self.reward = None

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None
    self.t += 1

    #self.action = self.controller.getMaxAction( [self.image, self.observation] )
    values = self.controller.getActionValues( [self.image, self.observation] )
    self.action = np.argmax(values, 1)

    return self.action, values

  def giveReward(self, r):
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    self.reward = r
    self.cumReward += sum(r)/float(len(r))
    for i in range(len(self.observation)):
      # Memory format: image box action reward time observations
      self.action[i] = self.observation[i].lastAction
      self.memory.append( [self.image] + self.observation[i].prevBox+['|']+ self.observation[i].nextBox + [self.action[i], self.reward[i], self.t, len(self.observation)] )

  def reset(self):
    self.observation = None
    self.action = None
    self.reward = None
    self.memory = []
    self.controller.loadNetwork()
    print 'Accumulated reward:',self.cumReward
    self.cumReward = 0

  def learn(self):
    if self.learner != None:
      self.learner.learn(self.memory, self.controller)

