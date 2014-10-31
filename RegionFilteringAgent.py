__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import RLConfig as config

import numpy as np
import scipy.io

class RegionFilteringAgent():

  image = None
  observation = None
  action = None
  reward = None
  memory = []
  timer = 0
  
  def __init__(self, qnet, learner=None, persistMemory=True):
    self.controller = qnet
    self.learner = learner
    self.avgReward = 0
    self.receivedRewards = 0
    self.persistMemory = persistMemory

  def integrateObservation(self, obs):
    if obs['image'] != self.image:
      self.observation = np.zeros( (2,obs['state'].shape[0]) )
      self.image = obs['image']
      self.timer = 0
    self.observation[1,:] = self.observation[0,:]
    self.observation[0,:] = obs['state']
    self.action = None
    self.reward = None
    print 'Agent::integrateObservation => image',self.image

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None

    self.timer += 1
    obs = self.observation.reshape( (1,self.observation.shape[0]*self.observation.shape[1]) )
    values = self.controller.getActionValues(obs)
    self.action = np.argmax(values, 1)[0]
    v = values[0,self.action]
    print 'Agent::getAction => action:',self.action, 'value:',v
    return (self.action,float(v))

  def giveReward(self, r):
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    obs = self.observation.reshape( (self.observation.shape[0]*self.observation.shape[1]))

    self.reward = r
    self.avgReward = (self.avgReward*self.receivedRewards + r)/(self.receivedRewards + 1)
    self.receivedRewards += 1
    self.memory.append( {'img':self.image, 't':self.timer, 'A':self.action, 'R':r, 'O':obs.tolist()} )
    print 'Agent::giveReward => ',r,self.avgReward

  def reset(self):
    print 'Agent::reset'
    self.image = ''
    self.observation = None
    self.action = None
    self.reward = None
    self.memory = []

  def learn(self):
    print 'Agent:learn:'
    self.saveMem()
    if self.learner != None:
      self.learner.learn(self.controller)
      self.controller.loadNetwork()

  def saveMem(self):
    if self.persistMemory:
      # Check that the memory belongs only to one image
      images = set([m['img'] for m in self.memory])
      assert len(images) == 1
      assert list(images)[0] == self.image

      # Sort records by time
      self.memory.sort(key=lambda x: x['t'])
      N = len(self.memory)
      # Collect records in matrices
      time = np.array([m['t'] for m in self.memory]).reshape((N,1))
      actions = np.array([m['A'] for m in self.memory]).reshape((N,1))
      rewards = np.array([m['R'] for m in self.memory]).reshape((N,1))
      observations = np.array([m['O'] for m in self.memory])
      # Save records to disk
      filename = config.get('trainMemory') + self.image + '.mat'
      contents = {'time':time, 'actions':actions, 'observations':observations, 'rewards':rewards}
      scipy.io.savemat(filename, contents, do_compression=True)

