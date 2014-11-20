__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import RLConfig as config

import numpy as np
import scipy.io
import MemoryUsage

import RLConfig as config

NUM_ACTIONS = config.geti('outputActions')

class BoxSearchAgent():

  image = None
  observation = None
  action = None
  reward = None
  timer = 0
  
  def __init__(self, qnet, learner=None):
    self.controller = qnet
    self.learner = learner
    self.avgReward = 0
    self.replayMemory = None

  def startReplayMemory(self, memoryImages, recordsPerImage, recordSize):
    self.replayMemory = ReplayMemory(memoryImages, recordsPerImage, recordSize)

  def integrateObservation(self, obs):
    if obs['image'] != self.image:
      self.actionsH = [0 for i in range(NUM_ACTIONS)]
      self.observation = np.zeros( (2,obs['state'].shape[0]), np.float32 )
      self.image = obs['image']
      self.timer = 0
      self.avgReward = 0.0
      if self.replayMemory != None:
        self.replayMemory.clear(self.image)
    self.observation[1,:] = self.observation[0,:]
    self.observation[0,:] = obs['state']
    self.action = None
    self.reward = None

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None

    self.timer += 1
    obs = self.observation.reshape( (1,self.observation.shape[0]*self.observation.shape[1]) )
    values = self.controller.getActionValues(obs)
    self.action = np.argmax(values, 1)[0]
    v = values[0,self.action]
    return (self.action,float(v))

  def giveReward(self, r):
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    self.reward = r
    self.avgReward = (self.avgReward*self.timer + r)/(self.timer + 1)
    if self.replayMemory != None:
      obs = self.observation.reshape( (self.observation.shape[0]*self.observation.shape[1]))
      self.replayMemory.add(self.image, self.timer, self.action, obs, self.reward)
    self.actionsH[self.action] += 1
    print 'Agent::MemoryRecord => image:',self.image,'time:',self.timer,'action:',self.action,'reward',self.reward,'avgReward:',self.avgReward

  def reset(self):
    print 'Agent::reset',self.actionsH,self.avgReward
    self.image = ''
    self.observation = None
    self.action = None
    self.reward = None

  def learn(self):
    print 'Agent:learn:'
    if self.learner != None and self.replayMemory != None:
      self.learner.learn(self.replayMemory, self.controller)
      self.controller.loadNetwork()

class ReplayMemory():

  def __init__(self, numImages, recordsPerImage, recordSize):
    self.O = np.zeros( (numImages*recordsPerImage, recordSize), np.float32 )
    self.A = np.zeros( (numImages*recordsPerImage, 1), np.int )
    self.R = np.zeros( (numImages*recordsPerImage, 1), np.float32 )
    self.numImages = numImages
    self.recordsPerImage = recordsPerImage
    self.index = {}
    self.pointer = 0

  def add(self, img, time, action, observation, reward):
    try: 
      key = self.index[img]
    except: 
      if self.pointer < self.O.shape[0]:
        self.index[img] = self.pointer
        self.pointer += self.recordsPerImage
        key = self.index[img]
      else:
        print 'Error: More images than expected!!'
    r = key + time - 1
    self.A[r,0] = action
    self.O[r,:] = observation
    self.R[r,0] = reward

  def clear(self, img):
    try:
      key = self.index[img]
    except:
      key = -1
    if key > 0:
      s = key
      e = key + self.recordsPerImage
      self.A[s:e,0] = 0
      self.O[s:e,:] = 0
      self.R[s:e,:] = 0

