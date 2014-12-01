__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

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

def tanh(x, a=5, b=0.5, c=2.0):
  return c*np.tanh(a*x + b)

class BoxSearchEnvironment(Environment, Named):

  def __init__(self, imageList, mode, groundTruthFile=None):
    self.mode = mode
    self.cnn = cn.ConvNet()
    self.testRecord = None
    self.idx = -1
    self.imageList = [x.strip() for x in open(imageList)]
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)
    #self.imageList = self.rankImages()
    #self.imageList = self.imageList[0:10]
    if self.mode == 'train':
      allImgs = set([x.strip() for x in open(config.get('allImagesList'))])
      self.negativeSamples = list(allImgs.difference(set(self.imageList)))
      self.negativeProbability = config.getf('negativeEpisodeProb')
      random.shuffle(self.imageList)
    self.loadNextEpisode()

  def performAction(self, action):
      self.state.performAction(action)

  def loadNextEpisode(self):
    self.episodeDone = False
    if self.selectNegativeSample(): return
    # Save actions performed during this episode
    if self.mode == 'test' and self.testRecord != None:
      with open(config.get('testMemory') + self.imageList[self.idx] + '.txt', 'w') as outfile:
        json.dump(self.testRecord, outfile)
    # Load a new episode
    self.idx += 1
    if self.idx < len(self.imageList):
      # Initialize state
      self.cnn.prepareImage(self.imageList[self.idx])
      self.state = bs.BoxSearchState(self.imageList[self.idx], groundTruth=self.groundTruth, randomStart=self.mode=='train')
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

  def selectNegativeSample(self):
    if self.mode == 'train' and random.random() < self.negativeProbability:
      idx = random.randint(0,len(self.negativeSamples)-1)
      self.cnn.prepareImage(self.negativeSamples[idx])
      self.state = bs.BoxSearchState(self.negativeSamples[idx], groundTruth=self.groundTruth, randomStart=True)
      print 'Environment::LoadNextEpisode => Random Negative:',idx,self.negativeSamples[idx]
      return True
    else:
      return False

  def updatePostReward(self, reward, allDone, cover):
    if self.mode == 'test':
      self.testRecord['boxes'].append( self.state.box )
      self.testRecord['actions'].append( self.state.actionChosen )
      self.testRecord['values'].append( self.state.actionValue )
      self.testRecord['rewards'].append( reward )
      self.testRecord['scores'].append( self.scores[:] )
      if self.state.actionChosen == bs.PLACE_LANDMARK:
        self.cnn.coverRegion(self.state.box)
        self.state.reset()
    elif self.mode == 'train':
      # We do not cover false landmarks during training
      if self.state.actionChosen == bs.PLACE_LANDMARK and cover:
        self.cnn.coverRegion(self.state.box) 
        self.state.reset()
      if allDone:
        self.episodeDone = True
    if self.state.actionChosen == bs.SKIP_REGION:
      self.state.reset()

  def getSensors(self):
    # Make a vector represenation of the action that brought the agent to this state (9 features)
    #prevAction = np.zeros( (bs.NUM_ACTIONS) )
    #prevAction[self.state.actionChosen] = 2.0 

    # Compute features of visible region and apply the sigmoid
    activations = self.cnn.getActivations(self.state.box)

    # Concatenate all info in the state representation vector
    #state = np.hstack( (visibleRegion, worldState, prevAction) )
    state = activations[config.get('convnetLayer')]
    self.scores = activations['prob'].tolist()
    return {'image':self.imageList[self.idx], 'state':state}

  def sampleAction(self):
    return self.state.sampleNextAction()
     
  def rankImages(self):
    keys = self.groundTruth.keys()
    keys.sort()
    # Rank by number of objects in the scene (from many to few)
    objectCounts = [len(self.groundTruth[k]) for k in keys]
    countRank = np.argsort(objectCounts)[::-1]
    countDist = dict([(i,0) for i in range(max(objectCounts)+1)])
    for o in objectCounts:
      countDist[o] += 1
    print 'Distribution of object counts (# objects vs # images):',countDist
    print 'Images with largest number of objects:',[keys[i] for i in countRank[0:10]]
    # Rank by object size (from small to large)
    minObjectArea = [ min(map(det.area, self.groundTruth[k])) for k in keys ]
    smallRank = np.argsort(minObjectArea)
    intervals = [ (500*400/i) for i in range(1,21) ]
    sizeDist = dict([ (i,0) for i in intervals ])
    for a in minObjectArea:
      counted = False
      for r in intervals:
        if a >= r: 
          sizeDist[r] += 1
          counted = True
          break
      if not counted: sizeDist[r] += 1
    print 'Distribution of smallest objects area (area vs # images):',[ (i,sizeDist[i]) for i in intervals]
    print 'Images with the smallest objects:',[keys[i] for i in smallRank[0:10]]
    # Rank by object size (from large to small)
    maxObjectArea = [ max(map(det.area, self.groundTruth[k])) for k in keys ]
    bigRank = np.argsort(minObjectArea)
    intervals = [ (500*400/i) for i in range(1,21) ]
    sizeDist = dict([ (i,0) for i in intervals ])
    for a in maxObjectArea:
      counted = False
      for r in intervals:
        if a >= r: 
          sizeDist[r] += 1
          counted = True
          break
      if not counted: sizeDist[r] += 1
    print 'Distribution of biggest objects area (area vs # images):',[ (i,sizeDist[i]) for i in intervals]
    print 'Images with the biggest objects:',[keys[i] for i in bigRank[0:10]]
    # Rank images by instance occlusion (from very occluded to isolated)
    maxInstanceOcclusion = []
    for k in keys:
      if len(self.groundTruth[k]) == 1:
        maxInstanceOcclusion.append(0)
      else:
        maxIoU = 0
        for i in range(len(self.groundTruth[k])):
          for j in range(i+1,len(self.groundTruth[k])):
            iou = det.IoU(self.groundTruth[k][i], self.groundTruth[k][j])
            if iou > maxIoU:
              maxIoU = iou
        maxInstanceOcclusion.append(maxIoU)
    occlusionRank = np.argsort(maxInstanceOcclusion)[::-1]
    intervals = [ 1.0/i for i in range(1,21) ]
    occlusionDist = dict([(i,0) for i in intervals])
    for o in maxInstanceOcclusion:
      counted = False
      for r in intervals:
        if o >= r:
          occlusionDist[r] += 1
          counted = True
          break
      if not counted: occlusionDist[r] += 1
    print 'Distribution of object occlusion (occlusion vs # images):',[(i,occlusionDist[i]) for i in intervals]
    print 'Images with the most occluded objects:',[keys[i] for i in occlusionRank[0:10]]
    # Rank combination
    rank = dict([(k,0) for k in keys])
    for i in range(len(keys)):
      rank[ keys[ countRank[i] ] ] += i
      rank[ keys[ smallRank[i]] ] += i
      rank[ keys[ occlusionRank[i] ] ] += i
    values = [ rank[i] for i in keys ]
    complexityRank = np.argsort(values)
    print 'More complex images:',[keys[i] for i in complexityRank[0:10]]
    return [keys[i] for i in smallRank]
    
