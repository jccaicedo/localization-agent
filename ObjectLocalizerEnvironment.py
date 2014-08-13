__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

from SingleObjectLocalizer import SingleObjectLocalizer

import random
import Image
import utils as cu

MAX_CANDIDATES_PER_IMAGE = 5

class ObjectLocalizerEnvironment(Environment, Named):
  
  def __init__(self, imgDir, candidatesFile, mode):
    self.imageDir = imgDir
    self.candidates = cu.loadBoxIndexFile(candidatesFile)
    self.imageIndex = self.candidates.keys()
    self.mode = mode
    self.terminalCounts = 0 
    if mode == 'Training':
      random.shuffle(self.imageIndex)
    else:
      self.imageIndex.sort()
    self.pointer = 0
    self.loadNextEpisode()

  def performAction(self, action):
    self.terminalCounts = 0
    for i in range(len(self.state)):
      prevAction = self.state[i].lastAction
      self.state[i].performAction( action[i] )
      if prevAction <= 1 or action[i] <= 1:
        self.terminalCounts += 1

  def updatePostReward(self):
    if len(self.state) == self.terminalCounts:
      self.pointer += 1
      self.loadNextEpisode()
      
  def getSensors(self):
    name = self.imageIndex[self.pointer]
    return (name, self.state)

  def loadNextEpisode(self):
    if self.pointer < len(self.imageIndex):
      name = self.imageIndex[self.pointer]
    elif self.mode == 'Training':
      random.shuffle(self.imageIndex)
      self.pointer = 0
    else:
      print 'All episodes done'
      return
    self.visibleImage = Image.open(self.imageDir + '/' + name + '.jpg')
    size = self.visibleImage.size
    self.state = [ SingleObjectLocalizer(size, box[0:4]) for box in self.candidates[name] ]
    if MAX_CANDIDATES_PER_IMAGE != 0:
      random.shuffle(self.state)
      self.state = self.state[0:MAX_CANDIDATES_PER_IMAGE]

