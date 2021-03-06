__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from RelationsDB import RelationsDB, CompactRelationsDB

import SimplifiedLayoutHandler as lh

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

class RegionFilteringEnvironment(Environment, Named):

  def __init__(self, featuresDir, mode):
    self.mode = mode
    self.testRecord = None
    if self.mode == 'train':
      self.db = CompactRelationsDB(featuresDir, randomize=True)
    elif self.mode == 'test':
      self.db = CompactRelationsDB(featuresDir, randomize=False)
    self.loadNextEpisode()

  def performAction(self, action):
      self.state.performAction(action)

  def loadNextEpisode(self):
    # Save actions performed during this episode
    if self.mode == 'test' and self.testRecord != None:
      with open(config.get('testMemory') + self.db.image + '.txt', 'w') as outfile:
        json.dump(self.testRecord, outfile)
    # Load a new episode from the database
    self.db.loadNext()
    if self.db.image != '':
      print 'Environment::LoadNextEpisode => Image',self.db.image,'({:4} boxes)'.format(self.db.boxes.shape[0])
      # Initialize state
      self.state = lh.LayoutHandler(self.db.boxes)
      self.performAction([lh.STAY,0]) 
    else:
      print 'No more images available'
    # Restart record for new episode
    if self.mode == 'test':
      self.testRecord = {'boxes':[], 'actions':[], 'values':[], 'rewards':[]}

  def updatePostReward(self, reward):
    if self.mode == 'test':
      self.testRecord['boxes'].append( self.db.boxes[self.state.selectedIds, :].tolist() )
      self.testRecord['actions'].append( self.state.actionChosen )
      self.testRecord['values'].append( self.state.actionValue )
      self.testRecord['rewards'].append( reward )

  def getSensors(self):
    # Create arrays to represent the state of the world
    worldState = self.state.getLocationState()

    # Make a vector represenation of the action that brought the agent to this state
    prevAction = np.zeros( (lh.NUM_ACTIONS) )
    prevAction[self.state.actionChosen] = 3.0 

    # Select features of visible regions and apply the sigmoid
    # Apply a threshold before transforming features?
    visibleRegions = np.copy(self.db.scores[ self.state.selectedIds, :])
    visibleRegions = tanh(visibleRegions) # (Greedy: not transformation required)

    # Pad zeros on features of void regions
    ## WARNING: Use zeros when using sigmoid. Use minus one for tanh.
    if len(self.state.selectedIds) == 0:
      visibleRegions = -1*np.ones( (lh.NUM_BOXES, self.db.scores.shape[1]) )
    elif len(self.state.selectedIds) < lh.NUM_BOXES:
      blankAreas = lh.NUM_BOXES - len(self.state.selectedIds)
      blanks = -1*np.ones( (blankAreas, visibleRegions.shape[1]) ) 
      visibleRegions = np.vstack( (visibleRegions, blanks) )
    visibleRegions = visibleRegions.reshape( (visibleRegions.shape[0]*visibleRegions.shape[1]) )

    # Concatenate all info in the state representation vector
    state = np.hstack( (visibleRegions, worldState, prevAction) )
    return {'image':self.db.image, 'state':state}
     
