__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from RelationsDB import RelationsDB

import LayoutHandler as lh

import random
import numpy as np

import utils as cu
import libDetection as det
import RLConfig as config

class RegionFilteringEnvironment(Environment, Named):

  def __init__(self, featuresDir, mode):
    self.mode = mode
    if self.mode == 'training':
      self.db = RelationsDB(featuresDir, randomize=False)
    else:
      self.db = RelationsDB(featuresDir, randomize=False)
    self.loadNextEpisode()

  def performAction(self, action):
    if self.mode == 'training':
      if self.state.percentExplored < 0.5:
        self.state.performAction(action)
      else:
        self.loadNextEpisode()
    else:
      if self.state.actionCounter < 30:
        self.state.performAction(action)
      else:
        self.loadNextEpisode()

  def loadNextEpisode(self):
    print 'Environment::LoadNextEpisode'
    self.db.loadNext()
    if self.db.image != '':
      print 'Image',self.db.image,'loaded in environment ({:4} boxes)'.format(self.db.boxes.shape[0])
      self.state = lh.LayoutHandler(self.db.boxes)
      self.performAction(0)
    else:
      print 'No more images available'

  def updatePostReward(self):
    pass

  def getSensors(self):
    # Build state representation:
    visibleRegions = self.db.scores[ self.state.selectedIds, :]
    worldExplored = np.asarray(self.state.status).reshape( (lh.WORLD_SIZE) )/float(self.db.boxes.shape[0])
    currentLocation = self.state.currentPosition.reshape( (lh.WORLD_SIZE) )
    prevAction = np.zeros( (lh.NUM_ACTIONS) )
    prevAction[self.state.actionChosen] = lh.NUM_ACTIONS
    if len(self.state.selectedIds) == 0:
      visibleRegions = -10*np.ones( (lh.NUM_BOXES, self.db.scores.shape[1]) )
    elif len(self.state.selectedIds) < lh.NUM_BOXES:
      blankAreas = lh.NUM_BOXES - len(self.state.selectedIds)
      blanks = -10*np.ones( (blankAreas, visibleRegions.shape[1]) )
      visibleRegions = np.vstack( (visibleRegions, blanks) )
    visibleRegions = visibleRegions.reshape( (visibleRegions.shape[0]*visibleRegions.shape[1]) )
    visibleRegions = 1/(1+np.exp(-visibleRegions))
    print 'State dimensionality:',visibleRegions.shape, worldExplored.shape, currentLocation.shape, prevAction.shape
    state = np.hstack( (visibleRegions, worldExplored, currentLocation, prevAction) )
    return {'image':self.db.image, 'state':state}
     
