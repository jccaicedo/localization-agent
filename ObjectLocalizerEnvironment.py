__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

from SingleObjectLocalizer import SingleObjectLocalizer

import random
import Image
import utils as cu

import RLConfig as config

import MemoryUsage

def loadBoxesFile(filename, threshold):
  gt = [x.split() for x in open(filename)]
  images = {}
  for k in gt:
    record = map(float,k[1:])
    if record[4] < threshold: continue
    try:
      images[k[0]] += [ map(float,k[1:]) ]
    except:
      images[k[0]] = [ map(float,k[1:]) ]
  return images

class ObjectLocalizerEnvironment(Environment, Named):
  
  def __init__(self, imgDir, candidatesFile, mode):
    self.imageDir = imgDir
    candidates = loadBoxesFile(candidatesFile, -2.0)
   
    self.mode = mode
    self.terminalCounts = 0
    self.episodeMoves = 0

    if mode == 'Training':
      self.balanceTrainingExamples(candidates)
    else:
      self.imageIndex = ImageBoxIndex(candidates, False)
    print 'Dataset ready:','{:5.2f}'.format(MemoryUsage.memory()/(1024**3)),'GB'
    self.loadNextEpisode()

  def performAction(self, action):
    #print 'ObjectLocalizerEnvironment::performAction(',action,')'
    self.terminalCounts = 0
    self.episodeMoves = 0
    for i in range(len(self.state)):
      prevAction = self.state[i].lastAction
      self.state[i].performAction( action[i] )
      if prevAction <= 1 or action[i] <= 1:
        self.terminalCounts += 1
      if len(self.state[0].history) > self.episodeMoves:
        self.episodeMoves = len(self.state[0].history)

  def updatePostReward(self):
    if len(self.state) == self.terminalCounts or self.episodeMoves >= config.MAX_MOVES_ALLOWED:
      self.loadNextEpisode()
      
  def getSensors(self):
    return (self.imgName, self.state)

  def loadNextEpisode(self):
    self.getExample()
    if self.imgName == None:
      print 'All episodes done'
      return

    self.visibleImage = Image.open(self.imageDir + '/' + self.imgName + '.jpg')
    size = self.visibleImage.size
    self.state = [ SingleObjectLocalizer(size, box[0:4]) for box in self.state ]
    if config.MAX_CANDIDATES_PER_IMAGE != 0:
      random.shuffle(self.state)
      self.state = self.state[0:config.MAX_CANDIDATES_PER_IMAGE]
    print 'New Episode:',self.imgName,'Boxes:',len(self.state),'Terminals:',self.terminalCounts,'Moves:',self.episodeMoves

  def balanceTrainingExamples(self, candidates):
    pos, neg = {},{}
    psc, ngc = 0,0
    for k in candidates.keys():
      for box in candidates[k]:
        if box[5] > config.MIN_POSITIVE_OVERLAP: # and box[6] >= 1.0:
          try: pos[k].append(box)
          except: pos[k] = [box]
          psc += 1
        else:
          try: neg[k].append(box)
          except: neg[k] = [box]
          ngc += 1

    self.pos = ImageBoxIndex(pos,True)
    self.neg = ImageBoxIndex(neg,True)
    self.probNeg = max(float(psc)/float(ngc),0.05)
    print 'Positives:',psc,'Negatives:',ngc

  def getExample(self):
    if self.mode == 'Training':
      if random.random() < self.probNeg:
        self.imgName, self.state = self.neg.getImage()
        self.neg.moveToNextImage()
      else:
        self.imgName, self.state = self.pos.getImage()
        self.pos.moveToNextImage()
    else:
      self.imgName, self.state = self.imageIndex.getImage()
      self.imageIndex.moveToNextImage()

class ImageBoxIndex():

  def __init__(self, data, doRand):
    self.data = data
    self.index = data.keys()
    self.pointer = 0
    self.doRand = doRand
    if doRand:
      random.shuffle(self.index)
    else:
      self.index.sort()

  def getImage(self):
    if self.pointer < len(self.index):
      name = self.index[self.pointer]
      return name, self.data[name]
    else:
      return None,None

  def moveToNextImage(self):
    if self.pointer+1 < len(self.index):
      self.pointer += 1
    elif self.doRand:
      self.pointer = 0
      random.shuffle(self.index)
    
