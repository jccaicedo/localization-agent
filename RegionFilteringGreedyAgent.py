__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import RLConfig as config

import numpy as np
import scipy.io
import MemoryUsage

import SimplifiedLayoutHandler as lh

THRESHOLD = -0.5

class RegionFilteringGreedyAgent():

  image = None
  observation = None
  action = None
  reward = None
  timer = 0
  
  def __init__(self, controller=None, learner=None):
    self.avgReward = 0
    self.scale = 0
    self.location = 0
    self.prevAction = 0
    self.action = None

  def startReplayMemory(self, memoryImages, recordsPerImage, recordSize):
    pass

  def integrateObservation(self, obs):
    if obs['image'] != self.image:
      self.actionsH = [0 for i in range(lh.NUM_ACTIONS)]
      self.image = obs['image']
      self.timer = 0
      self.avgReward = 0.0
      self.scale = 0
      self.location = 0
      self.direction = 1
      self.spaceDirection = 1
    self.prevAction = self.action
    self.observation = np.copy(obs['state'])
    self.action = None
    self.reward = None

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None
    self.timer += 1

    # Read the current position and state
    scale = np.argmax(self.observation[189:199])
    location = np.argmax(self.observation[199:208])
    A = self.observation[0:180]
    detectorsResponse = [A[1], A[61], A[121]]
    # Select action using predefined rules
    #print 'FEATURES',detectorsResponse
    print scale,location
    if np.max(detectorsResponse) > THRESHOLD:
      # Explore cell further if response above threshold
      self.action = lh.STAY
    else:
      # Continue world exploration
      ####################################
      #  0 3 6  #  This is the indexing
      #  1 4 7  #  of the world according
      #  2 5 8  #  to the LayoutHandler
      ####################################
      if self.spaceDirection == 1 and location < 8:
        possibleActions = {0:lh.GO_RIGHT, 1:lh.GO_BACK, 2:lh.GO_RIGHT, 3:lh.GO_RIGHT, 4:lh.GO_LEFT, 5:lh.GO_RIGHT, 6:lh.GO_BACK, 7:lh.GO_LEFT}
        self.action = possibleActions[location] 
      elif self.spaceDirection == -1 and location > 0:
        possibleActions = {1:lh.GO_RIGHT, 2:lh.GO_FRONT, 3:lh.GO_LEFT, 4:lh.GO_RIGHT, 5:lh.GO_LEFT, 6:lh.GO_LEFT, 7:lh.GO_FRONT, 8:lh.GO_LEFT}
        self.action = possibleActions[location]
      elif location == 8 or location == 0:
        if self.direction == 1:
          self.action = lh.GO_DOWN
        elif self.direction == -1:
          self.action = lh.GO_UP
      else:
        print 'ERROR: Unknown state current[{:1},{:1}] past[{:1},{:1}] A={:1}'.format(scale,location,self.scale,self.location,self.prevAction)
        self.action = -1
    # Chose direction of scale exploration
    if scale == 9:
      self.direction = -1
    elif scale == 0:
      self.direction = 1
    # Chose direction of space exploration
    if location == 8:
      self.spaceDirection = -1
    elif location == 0:
      self.spaceDirection = 1
    # Save current observed location
    self.scale = scale
    self.location = location
    return (self.action,1.0)

  def giveReward(self, r):
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    self.reward = r
    self.avgReward = (self.avgReward*self.timer + r)/(self.timer + 1)
    self.actionsH[self.action] += 1
    print 'Agent::MemoryRecord => image:',self.image,'time:',self.timer,'action:',self.action,'reward',self.reward,'avgReward:',self.avgReward

  def reset(self):
    print 'Agent::reset',self.actionsH,self.avgReward
    self.image = ''
    self.observation = None
    self.action = None
    self.reward = None

  def learn(self):
    pass


