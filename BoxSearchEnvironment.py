__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from RelationsDB import RelationsDB, CompactRelationsDB

import BoxSearchState as bs
import ConvNet as cn

import random
import numpy as np
import json

import utils as cu
import libDetection as det
import RLConfig as config

def sigmoid(x, a=1.0, b=0.0):
  return 1.0/(1.0 + np.exp(-a*x + b))

def tanh(x, a=0.75, b=0.25):
  return np.tanh(a*x + b)

class BoxSearchEnvironment(Environment, Named):

  def __init__(self, imageList, mode):
    self.mode = mode
    self.cnn = cn.ConvNet()
    self.testRecord = None
    self.idx = -1
    self.imageList = [x.strip() for x in open(imageList)]
    if self.mode == 'train':
      random.shuffle(self.imageList)
    self.loadNextEpisode()

  def performAction(self, action):
      self.state.performAction(action)

  def loadNextEpisode(self):
    # Save actions performed during this episode
    if self.mode == 'test' and self.testRecord != None:
      with open(config.get('testMemory') + self.imageList[self.idx] + '.txt', 'w') as outfile:
        json.dump(self.testRecord, outfile)
    # Load a new episode
    self.idx += 1
    if self.idx < len(self.imageList):
      # Initialize state
      self.cnn.prepareImage(self.imageList[self.idx])
      self.state = bs.BoxSearchState(self.imageList[self.idx], self.mode == 'train')
      print 'Environment::LoadNextEpisode => Image',self.idx,self.imageList[self.idx],'('+str(self.state.visibleImage.size[0])+','+str(self.state.visibleImage.size[1])+')'
    else:
      if self.mode == 'train':
        random.shuffle(self.imageList)
        self.idx = -1
        self.loadNextEpisode()
      else:
        print 'No more images available'
    # Restart record for new episode
    if self.mode == 'test':
      self.testRecord = {'boxes':[], 'actions':[], 'values':[], 'rewards':[], 'scores':[]}

  def updatePostReward(self, reward):
    if self.mode == 'test':
      self.testRecord['boxes'].append( self.state.box )
      self.testRecord['actions'].append( self.state.actionChosen )
      self.testRecord['values'].append( self.state.actionValue )
      self.testRecord['rewards'].append( reward )
      self.testRecord['scores'].append( self.scores[:] )

  def getSensors(self):
    # Create arrays to represent the state of the world (17 features)
    worldState = self.state.getRepresentation()
    worldState = np.array( worldState )

    # Make a vector represenation of the action that brought the agent to this state (8 features)
    prevAction = np.zeros( (bs.NUM_ACTIONS) )
    prevAction[self.state.actionChosen] = 1.0 

    # Compute features of visible region and apply the sigmoid
    visibleRegion = self.cnn.getActivations(self.state.box)

    # Concatenate all info in the state representation vector
    state = np.hstack( (visibleRegion, worldState, prevAction) )
    self.scores = visibleRegion.tolist()
    return {'image':self.imageList[self.idx], 'state':state}
     
