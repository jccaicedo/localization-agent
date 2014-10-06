__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

from SingleObjectLocalizer import SingleObjectLocalizer

import random
import Image
import utils as cu

import RLConfig as config

import MemoryUsage
import copy

def loadBoxesFile(filename, threshold):
  gt = [x.split() for x in open(filename)]
  images = {}
  for k in gt:
    record = map(float,k[1:])
    if record[4] < threshold: continue
    try:
      images[k[0]] += [ record ]
    except:
      images[k[0]] = [ record ]
  return images

class ObjectLocalizerEnvironment(Environment, Named):
  
  def __init__(self, imgDir, candidatesFile, mode):
    self.imageDir = imgDir
   
    self.mode = mode
    self.terminalCounts = 0
    self.episodeMoves = 0
    self.imgName = None
    self.state = None
    self.moreEpisodes = True

    if mode == 'Training':
      candidates = loadBoxesFile(candidatesFile, -2.0)
      self.balanceTrainingExamples(candidates)
    else:
      candidates = loadBoxesFile(candidatesFile, -10.0)
      self.imageIndex = ImageBoxIndex(candidates, False)
    print 'Dataset ready:','{:5.2f}'.format(MemoryUsage.memory()/(1024**3)),'GB'
    self.loadNextEpisode()

  def performAction(self, actions):
    action = actions[0]
    values = actions[1]
    self.terminalCounts = 0
    self.episodeMoves = 0
    for i in range(len(self.state)):
      prevAction = self.state[i].lastAction
      self.state[i].performAction( action[i], values[i,:] )
      if prevAction <= 1 or action[i] <= 1:
        self.terminalCounts += 1
      if len(self.state[i].history) > self.episodeMoves:
        self.episodeMoves = len(self.state[i].history)

  def updatePostReward(self):
    if len(self.state) == self.terminalCounts or self.episodeMoves >= config.geti('maxMovesAllowed'):
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
    self.state = [ SingleObjectLocalizer(size, box[0:4], box[4]) for box in self.state ]
    if config.geti('maxCandidatesPerImage') != 0 and self.mode == "Training":
      random.shuffle(self.state)
      self.state = self.state[0: config.geti('maxCandidatesPerImage')]
    print 'Episode done:',self.imgName,'Boxes:',len(self.state),'Terminals:',self.terminalCounts,'Moves:',self.episodeMoves

  def balanceTrainingExamples(self, candidates):
    pos, neg = {},{}
    psc, ngc = 0,0
    for k in candidates.keys():
      for box in candidates[k]:
        if box[5] > config.getf('minPositiveOverlap'): # and box[6] >= 1.0:
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
      self.imageIndex.recordState( self.imgName, self.state )
      self.imgName, self.state = self.imageIndex.getImage()
      self.moreEpisodes = self.imageIndex.moveToNextImage()

  def hasMoreEpisodes(self):
      return self.moreEpisodes

  def saveRecords(self, outputFile):
    if self.mode != 'Training':
      self.imageIndex.saveRecords(outputFile)
    
class ImageBoxIndex():

  def __init__(self, data, doRand):
    self.data = data
    self.index = data.keys()
    self.pointer = -1
    self.doRand = doRand
    if doRand:
      random.shuffle(self.index)
    else:
      self.index.sort()
    self.terminalStates = {}

  def getImage(self):
    if self.pointer < len(self.index):
      name = self.index[self.pointer]
      return name, copy.deepcopy(self.data[name])
    else:
      return None,None

  def moveToNextImage(self):
    if self.pointer+1 < len(self.index):
      self.pointer += 1
    elif self.doRand:
      self.pointer = 0
      random.shuffle(self.index)
    else:
      return False
    return True

  def recordState(self, img, state):
    if state == None: return
    rec = {'boxes':[], 'history':[]}
    for i in range(len(state)):
      finalBox = map(int, state[i].nextBox)
      rec['boxes'].append( finalBox + [state[i].terminalScore] )
      rec['history'].append( self.data[img][i] + state[i].history)
    self.terminalStates[img] = rec 

  def saveRecords(self, outputFile):
    out1 = open(outputFile, 'w')
    out2 = open(outputFile+'.moves', 'w')
    for img in self.terminalStates.keys():
      for k in range(len(self.terminalStates)):
        box = self.terminalStates[img]['boxes'][k]
        hist = self.terminalStates[img]['history'][k]
        finalBox = ' '.join(map(str, box[0:4]))
        out1.write(img + ' ' + str(box[-1]) + ' ' + finalBox + ' 0\n')
        out2.write(img + ' ' + ' '.join(map(str, map(int, hist[0:4]))) + ' ' 
                       + ' '.join(map(str,hist[4:6])) + ' ' + ' '.join(map(str,hist[7:]))
                       + ' ' + finalBox + ' ' + str(box[-1]) + '\n')
    out1.close()
    out2.close()

