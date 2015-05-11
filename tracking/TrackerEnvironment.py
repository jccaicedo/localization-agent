__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

import TrackerState as ts
import ConvNet as cn
import sequence

import random
import numpy as np
import json
import os

import utils.utils as cu
import utils.libDetection as det
import learn.rl.RLConfig as config

def sigmoid(x, a=1.0, b=0.0):
  return 1.0/(1.0 + np.exp(-a*x + b))

def tanh(x, a=5, b=0.5, c=2.0):
  return c*np.tanh(a*x + b)

def selectInitBox(imageKey, groundTruth):
    tokens = imageKey.split('/')
    sequenceName = tokens[0]
    imageName = tokens[-1]
    previousImageName = os.path.join(sequenceName, tokens[1], '{:04d}'.format(int(imageName)-1))
    episodeMemPath = os.path.join(config.get('testMemory'), imageKey + '.txt')
    #TODO: use a better way to know to track starting bbox
    if os.path.exists(episodeMemPath):
        testMemory = cu.load_memory(episodeMemPath)
        if ts.PLACE_LANDMARK in testMemory['actions']:
            landmarkIndex = testMemory['actions'].index(ts.PLACE_LANDMARK)
            return testMemory['boxes'][landmarkIndex]
        else:
            return selectInitBox(previousImageName, groundTruth)
    elif imageKey in groundTruth:
        return groundTruth[imageKey][0]
    else:
        raise Exception('Unexpected condition for image key {}'.format(imageKey))

TEST_TIME_OUT = config.geti('testTimeOut')

class TrackerEnvironment(Environment, Named):

  def __init__(self, sequenceDatabase, mode):
    self.mode = mode
    self.cnn = cn.ConvNet()
    self.testRecord = None
    self.idx = -1
    self.sequenceDatabase = [x.strip() for x in open(sequenceDatabase)]
    self.sequenceSpecs = [cu.parseSequenceSpec(aSequenceSpec) for aSequenceSpec in self.sequenceDatabase]
    self.imageList = []
    self.imageSuffix = config.get('frameSuffix')
    self.sequenceDir = config.get('sequenceDir')
    self.groundTruth = {}
    #TODO: how to handle duplicates, etc
    for i in range(len(self.sequenceSpecs)):
        seqName, seqSpan, seqStart, seqEnd = self.sequenceSpecs[i]
        imageDir = os.path.join(self.sequenceDir, seqName, config.get('imageDir'))
        gtPath = os.path.join(self.sequenceDir, seqName, config.get('gtFile'))
        aSequence = sequence.fromdir(imageDir, gtPath, suffix=self.imageSuffix)
        #no seqSpan means full sequence
        #frames start at 2 in list, but include 0 in gt
        if seqSpan is None:
            start = 1
            end = len(aSequence.frames)
        else:
            start = int(seqStart)
            end = int(seqEnd)
            if start < 1 or end > len(aSequence.frames) or start >= end:
                raise ValueError('Start {} or end {} outside of bounds {},{}'.format(start, end, 1, len(aSequence.frames)))
        for j in range(start, end+1):
            imageKey = os.path.join(seqName, config.get('imageDir'), aSequence.frames[j-1])
            if j > start:
                self.imageList.append(imageKey)
            self.groundTruth[imageKey] = [aSequence.boxes[j-1].tolist()]
    if self.mode == 'train':
      random.shuffle(self.imageList)
    self.loadNextEpisode()

  def performAction(self, action):
      self.state.performAction(action)

  def loadNextEpisode(self):
    self.episodeDone = False
    # Save actions performed during this episode
    if self.mode == 'test' and self.testRecord != None:
      episodeMemPath = os.path.join(config.get('testMemory'), self.imageList[self.idx] + '.txt')
      episodeMemDir = os.path.dirname(episodeMemPath)
      if not os.path.exists(episodeMemDir):
          os.makedirs(episodeMemDir)
      with open(episodeMemPath, 'w') as outfile:
        json.dump(self.testRecord, outfile)
    # Load a new episode
    self.idx += 1
    if self.idx < len(self.imageList):
      # Initialize state
      tokens = self.imageList[self.idx].split('/')
      sequenceName = tokens[0]
      imageName = tokens[-1]
      previousImageName = os.path.join(sequenceName, tokens[1], '{:04d}'.format(int(imageName)-1))
      print 'Preparing starting image {}'.format(previousImageName)
      self.cnn.prepareImage(previousImageName)
      if self.mode == 'test':
        initialBox = selectInitBox(previousImageName, self.groundTruth)
      else:
        initialBox = self.groundTruth[previousImageName][0]
      print 'Initial box for {} at {}'.format(previousImageName, initialBox)
      self.startingActivations = self.cnn.getActivations( initialBox)
      self.cnn.prepareImage(self.imageList[self.idx])
      self.state = ts.TrackerState(self.imageList[self.idx], self.mode, groundTruth=self.groundTruth)
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

  def updatePostReward(self, reward, allDone, cover):
    if self.mode == 'test':
      self.testRecord['boxes'].append( self.state.box )
      self.testRecord['actions'].append( self.state.actionChosen )
      self.testRecord['values'].append( self.state.actionValue )
      self.testRecord['rewards'].append( reward )
      self.testRecord['scores'].append( self.scores[:] )
      if self.state.actionChosen == ts.PLACE_LANDMARK:
        self.state.reset()
      if self.state.stepsWithoutLandmark > TEST_TIME_OUT:
        self.state.reset()
    elif self.mode == 'train':
      if self.state.actionChosen == ts.PLACE_LANDMARK and len(cover) > 0:
        self.state.reset()
      if allDone:
        self.episodeDone = True
    # Terminate episode with a single detected instance
    if self.state.actionChosen == ts.PLACE_LANDMARK:
      self.episodeDone = True

  def getSensors(self):
    # Make a vector represenation of the action that brought the agent to this state (9 features)
    prevAction = np.zeros( (ts.NUM_ACTIONS) )
    prevAction[self.state.actionChosen] = 1.0

    # Compute features of visible region (4096 + 21)
    activations = self.cnn.getActivations(self.state.box)

    # Concatenate all info in the state representation vector
    state = np.hstack( (activations[config.get('convnetLayer')], self.startingActivations[config.get('convnetLayer')], prevAction) )
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
    return [keys[i] for i in occlusionRank]
    
