__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from LayoutHandler import LayoutHandler

import random
import numpy as np

import utils as cu
import libDetection as det
import RLConfig as config

class RegionFilteringEnvironment(Environment, Named):
  def __init__(self, listOfImages, featuresDir, mode):
    self.images = [x.replace('\n','') for x in open(listOfImages)]
    self.featuresDir = featuresDir
    self.mode = mode
    self.next = 0
    self.features = None
    self.loadNextEpisode()

  def performAction(self, action):
    if action == 'terminate':
      print 'Agent terminates current episode'
      self.loadNextEpisode()
      return
    area, box = self.state.getBoxesInfo( [action[0]] )
    self.state.selectBox( action[0] )
    indexes = self.state.performAction( action[1] )
    if indexes != None and len(indexes) > 0:
      print self.steps, 'RegionFilteringEnvironment::performAction::act,img,idxs',action[0:2], self.images[self.next-1], indexes
      areas, boxes = self.state.getBoxesInfo( indexes )
      self.observation = {'indexes':indexes, 'areas':areas, 'boxes':boxes, 'features':self.getFeatures(indexes), 'imgSize':self.state.frame[2:],
                          'lastAction':action[1], 'rootBox':box[0], 'rootArea':area, 'rootFeatures':self.getFeatures([action[0]])}
      self.steps += 1
    else:
      self.observation = {"NoChange":True, 'rootBox':box[0], 'lastAction':action[1], 'boxes':[]}
      print 'Environment::performAction => NoChange',action[0:2]
    if action[0] == -1:
      print 'Initializing new image'
      self.observation['NewImage'] = True


  def loadNextEpisode(self):
    print 'Environment::LoadNextEpisode'
    if self.mode == 'Training':
      if self.next == len(self.images):
        random.shuffle(self.images)
        self.next = 0
      sourceFile = self.featuresDir + self.images[self.next] + '.sigmoid_scores'
      print 'Loading:',sourceFile,'in environment'
      features,boxes = cu.loadMatrixAndIndex(sourceFile)
      self.state = LayoutHandler(boxes)
      self.features = features
      print 'Image',self.images[self.next],'boxes:',self.features.shape
      self.steps = 0
      self.performAction([-1,9])
      self.next += 1

  def updatePostReward(self):
    if self.steps >= 20: #config.MAX_STEPS_ALLOWED:
      self.loadNextEpisode()

  def getSensors(self):
    return (self.images[self.next-1], self.observation)

  def getFeatures(self, indexes):
    if self.mode == 'Training':
      rows = [i for i in indexes if i < self.features.shape[0]]
      if len(rows) > 0:
        return self.features[ rows, :]
      else:
        print 'INITIALIZING FEATURES FOR',indexes,'WITH',self.features.shape[1],'ZEROS'
        return np.zeros( (len(indexes), self.features.shape[1]) )

