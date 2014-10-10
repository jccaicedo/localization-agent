__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

import random

import utils as cu
import libDetection as det
import RLConfig as config

class RegionFilteringEnvironment(Environment, Named):
  def __init__(self, listOfImages, featuresDir):
    self.images = [x.replace('\n','') for x in open(listOfImages)]
    self.featuresDir = featuresDir
    self.mode = 'Training'
    self.next = len(self.images)
    self.features = None

  def performAction(self, action):
    self.state.selectBox( action[0] )
    indexes = self.state.performAction( action[1] )
    areas, boxes = self.state.getBoxesInfo( indexes )
    self.observation = {'indexes':indexes, 'areas':areas, 'boxes':boxes, 'features':self.getFeatures(indexes)}
    self.steps += 1

  def loadNextEpisode(self):
    if self.mode == 'Training':
      if self.next == len(self.images):
        random.shuffle(self.images)
        self.next = -1
      self.next += 1
      features,boxes = cu.loadMatrixAndIndex(self.images[self.next])
      self.state = LayoutHandler(boxes)
      self.features = features
      self.steps = 0

  def updatePostReward(self):
    if self.steps >= config.MAX_STEPS_ALLOWED:
      self.loadNextEpisode()

  def getSensors(self):
    return self.observation

  def getFeatures(self, indexes):
    if self.mode == 'Training':
      return self.features[ indexes, :]
