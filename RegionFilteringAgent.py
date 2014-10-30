__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import RLConfig as config

import numpy as np

class RegionFilteringAgent():

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
    if obs['image'] != self.image:
      self.observation = np.zeros( (2,obs['state'].shape[0]) )
      self.image = obs['image']
    self.observation[1,:] = self.observation[0,:]
    self.observation[0,:] = obs['state']
    self.action = None
    self.reward = None
    print 'Agent::integrateObservation'

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None
    self.t += 1

    print 'Agent::getAction'

    obs = self.observation.reshape( (1,self.observation.shape[0]*self.observation.shape[1]) )
    values = self.controller.getActionValues(obs)
    self.action = np.argmax(values, 1)

    return self.action

  def giveReward(self, r):
    print 'Agent::giveReward=>',r
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    self.reward = r
    self.cumReward += r

    if self.action != 'terminate':
      obs = self.observation.reshape( (self.observation.shape[0]*self.observation.shape[1]))
      self.memory.append( {'A':self.action, 'R':r, 'O':obs} )

  def reset(self):
    print 'Agent::reset'
    self.image = ''
    self.observation = None
    self.action = None
    self.reward = None
    self.memory = []
    self.controller.loadNetwork()
    self.t = 0

  def learn(self):
    print 'Agent:learn:'
    if self.learner != None:
      self.learner.learn(self.memory, self.controller)

