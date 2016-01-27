__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

import TrackerState as ts
import ConvNet as cn
import sequence
import TrajectorySimulator as tsim

import random
import numpy as np
import json
import os, sys

import utils.utils as cu
import utils.libDetection as det
import benchmarkUtils as benchutils
import learn.rl.RLConfig as config

def sigmoid(x, a=1.0, b=0.0):
  return 1.0/(1.0 + np.exp(-a*x + b))

def tanh(x, a=5, b=0.5, c=2.0):
  return c*np.tanh(a*x + b)

TEST_TIME_OUT = config.geti('testTimeOut')

class TrackerAugmentedEnvironment(Environment, Named):

  def __init__(self, sequenceDatabase, mode):
    self.mode = mode
    self.cnn = cn.ConvNet()
    self.testRecord = None
    self.frameIdx = -1
    self.episodeIdx = -1
    self.sequenceDatabase = [x.strip() for x in open(sequenceDatabase)]
    self.sequenceSpecs = [cu.parseSequenceSpec(aSequenceSpec) for aSequenceSpec in self.sequenceDatabase]
    self.imageSuffix = config.get('frameSuffix')
    self.sequenceDir = config.get('sequenceDir')
    self.numEpisodesPerEpoch = config.geti('trainingEpisodesPerEpoch')
    self.episodeCounter = 0

    #TODO: how to handle duplicates, etc
    self.episodes = []
    for i in range(len(self.sequenceSpecs)):
      e = TrackingEpisode(self.sequenceSpecs[i], self.sequenceDir, self.imageSuffix)
      self.episodes.append(e)

    if mode == 'train':
      self.objectBoxes = cu.loadBoxIndexFile(config.get('objectBoxesFile'))
      self.objectImageDir = config.get('objectImageDir')
      #self.generateRandomSequence()

    self.loadNextEpisode()

  def generateRandomSequence(self):
    sequenceDir = self.sequenceDir + '/' + self.episodes[0].seqName # change 0 for current sequence pointer?
    scene = np.random.randint(len(self.objectBoxes.keys()))
    obj = np.random.randint(len(self.objectBoxes.keys()))
    #TODO: change box to polygon
    simulator = tsim.TrajectorySimulator(self.objectImageDir + '/' + self.objectBoxes.keys()[scene] + '.jpg', 
                            self.objectImageDir + '/' + self.objectBoxes.keys()[obj] + '.jpg' , 
                            map(int, self.objectBoxes[self.objectBoxes.keys()[obj]][0]) )
    while simulator.nextStep(): 
      simulator.saveFrame(sequenceDir)

  def performAction(self, action):
      self.state.performAction(action)

  def loadNextFrame(self):
    if self.frameIdx > 0 and not self.landmarkFound:
      self.episodes[self.episodeIdx].notifyFrameDone(self.frameIdx, None, self.mode)
    self.landmarkFound = False
    self.frameIdx += 1
    # Save actions performed in this frame
    if self.mode == 'test' and self.testRecord != None:
      episodeMemPath = os.path.join(config.get('testMemory'), self.currentFrameName() + '.txt')
      episodeMemDir = os.path.dirname(episodeMemPath)
      if not os.path.exists(episodeMemDir):
          os.makedirs(episodeMemDir)
      with open(episodeMemPath, 'w') as outfile:
        json.dump(self.testRecord, outfile)
    if self.mode == 'train' and self.testRecord != None:
      self.episodes[self.episodeIdx].saveRecord(self.frameIdx-1, self.testRecord)
    # Initialize state
    self.cnn.prepareImage(self.currentFrameName())
    initialBox = self.episodes[self.episodeIdx].getNextInitialBox()
    self.state = ts.TrackerState(self.currentFrameName(), self.mode, initialBox['box'], groundTruth=self.getGroundTruth())
    w,h = self.state.visibleImage.size
    print 'Environment::LoadNextFrame => Image',self.frameIdx,self.currentFrameName(),'('+str(w)+','+str(h)+')@',initialBox
    # Restart record for new episode
    #if self.mode == 'test':
    self.testRecord = {'boxes':[], 'actions':[], 'values':[], 'rewards':[]}
    if self.frameIdx == self.numFrames()-1:
      self.episodeDone = True

  def loadNextEpisode(self):
    self.episodeDone = False
    self.frameIdx = -1
    if self.mode == 'train':
      self.episodes[self.episodeIdx].dumpLog()
      self.episodeIdx = 0
      if self.episodeCounter < self.numEpisodesPerEpoch:
        self.episodeCounter += 1
        self.generateRandomSequence()
        self.episodes[self.episodeIdx].reloadGroundTruths()
        self.episodes[self.episodeIdx].cleanHistory()
      else:
        print 'All training episodes done'
        return
    else:
      if self.episodeIdx < len(self.episodes)-1:
        self.episodes[self.episodeIdx].cleanHistory()
        self.episodeIdx += 1
      else:
        print 'All test episodes done'
        return
    print '***********************************************************'
    print 'Environment::LoadNextEpisode => Frames:',self.numFrames(),'Boxes:',len(self.getGroundTruth())
    print '***********************************************************'
    self.episodes[self.episodeIdx].setupTargetObject(self.cnn)

  def numEpisodes(self):
    # One episode is one video sequence
    if self.mode == "train":
      return self.numEpisodesPerEpoch
    elif self.mode == "test":
      return len(self.episodes)

  def numFrames(self):
    return len(self.episodes[self.episodeIdx].imageList)

  def getGroundTruth(self):
    return self.episodes[self.episodeIdx].groundTruth

  def currentFrameName(self):
    return self.episodes[self.episodeIdx].imageList[self.frameIdx]

  def currentFrameGroundTruth(self):
    return self.getGroundTruth()[self.currentFrameName()]

  def epochDone(self):
    self.episodeCounter = 0
    self.episodeIdx = -1

  def updatePostReward(self, reward):
    # Save interaction info
    if True: #self.mode == 'test':
      self.testRecord['boxes'].append( self.state.box )
      self.testRecord['actions'].append( self.state.actionChosen )
      self.testRecord['values'].append( self.state.actionValue )
      self.testRecord['rewards'].append( reward )
    # Terminate episode with a single detected instance
    if self.state.actionChosen == ts.PLACE_LANDMARK:
      self.landmarkFound = True
      self.episodes[self.episodeIdx].notifyFrameDone(self.frameIdx, self.state.box, self.mode)

  def getSensors(self):
    # Compute features of visible region
    activations = self.cnn.getActivations(self.state.box)
    state = self.episodes[self.episodeIdx].computeSimilarities(activations[config.get('convnetLayer')])
    return {'image':self.currentFrameName(), 'state':state}

  def sampleAction(self):
    return self.state.sampleNextAction()

######################################
# DISTANCE AND SIMILARITY FUNCTIONS
######################################

MAP_SHAPE = map(int, config.get('featureMapShape').split(','))

def cosine(A,B):
  return np.sum( np.multiply(A,B) ) / (np.linalg.norm(A)*np.linalg.norm(B) )

def euclidean(A,B):
  return np.sqrt( np.sum( (A - B)**2 ) )

def slidingWindowIndex(dim, minLength):
  a = range(dim-1, -1, -1) + [0 for i in range(dim-1)]
  b = [dim for i in range(dim-1)] + range(dim, 0, -1)
  c = [0 for i in range(dim-1)] + range(dim)
  d = range(1,dim) + [dim for i in range(dim)]
  full = zip(a,b,c,d)
  idx = []
  for t in full:
    if t[1]-t[0] >= minLength:
      idx.append(t)
  return idx
 
SWI = slidingWindowIndex(MAP_SHAPE[-1], 3) 
FEATURE_DIM = len(SWI)**2

def slide(A,B):
  C = np.zeros((len(SWI)**2))
  r = 0
  for s in SWI:
    for t in SWI:
      C[r] = cosine(A[:, s[2]:s[3], t[2]:t[3]] , B[:, s[0]:s[1], t[0]:t[1]])
      r += 1
  return C

TRACKING_HISTORY = config.geti('similarityTrackingHistory')
FEATURE_MAP_DIMS = config.geti('featureMapDimensions')

######################################
# TRACKING EPISODE CLASS
######################################

class TrackingEpisode():

  def __init__(self, sequenceSpecs, sequenceDir, imageSuffix):
    # Basic sequence info
    self.imageList = []
    self.groundTruth = {}
    self.seqName, self.seqSpan, self.seqStart, self.seqEnd = sequenceSpecs
    imageDir = os.path.join(sequenceDir, self.seqName, config.get('imageDir'))
    self.gtPath = os.path.join(sequenceDir, self.seqName, config.get('gtFile'))
    aSequence = sequence.fromdir(imageDir, self.gtPath, suffix=imageSuffix)
    #no seqSpan means full sequence
    #frames start at 1 in list, but include 0 in gt
    if self.seqSpan is None:
      start = 1
      end = len(aSequence.frames)
    else:
      start = int(seqStart)
      end = int(seqEnd)
      if start < 1 or end > len(aSequence.frames) or start >= end:
          raise ValueError('Start {} or end {} outside of bounds {},{}'.format(start, end, 1, len(aSequence.frames)))
    for j in range(start, end+1):
      imageKey = os.path.join(self.seqName, config.get('imageDir'), aSequence.frames[j-1])
      if j > start:
        self.imageList.append(imageKey)
      self.groundTruth[imageKey] = [aSequence.boxes[j-1].tolist()]
    # Sequence history objects
    self.landmarkFeatures = None
    self.landmarkBoxes = []
    self.lastBoxFeatures = None
    self.boxLog = []

  def reloadGroundTruths(self):
    boxes = benchutils.parse_gt(self.gtPath)
    j = 0
    for key in self.groundTruth.keys():
      self.groundTruth[key] = [boxes[j].tolist()]
      j += 1

  def setupTargetObject(self, cnn):
    # Initialize memory
    self.landmarkFeatures = np.zeros( [TRACKING_HISTORY]+list(MAP_SHAPE) )
    # Find first frame and its object bounding box
    tokens = self.imageList[0].split('/')
    sequenceName = tokens[0]
    imageName = tokens[-1]
    firstImageName = os.path.join(sequenceName, tokens[1], '{:04d}'.format(int(imageName)-1))
    self.targetObjectBox = self.groundTruth[firstImageName][0]
    # Compute features for the object
    cnn.prepareImage(firstImageName)
    self.targetObjectFeatures = cnn.getActivations(self.targetObjectBox)
    self.targetObjectFeatures = np.resize(self.targetObjectFeatures[config.get('convnetLayer')], MAP_SHAPE)
    print 'Target Object set for {} at {}'.format(firstImageName, self.targetObjectBox)

  def cleanHistory(self):
    self.landmarkFeatures = None
    self.landmarkBoxes = []
    self.lastBoxFeatures = None
    self.targetObjectFeatures = None

  def computeSimilarities(self, features):
    self.lastBoxFeatures = features
    R = np.zeros( ((TRACKING_HISTORY+1)*FEATURE_DIM) )
    R[0:FEATURE_DIM] = slide(features, self.targetObjectFeatures)
    for i in range(TRACKING_HISTORY):
      R[FEATURE_DIM*(i+1):FEATURE_DIM*(i+2)] = slide(features, self.landmarkFeatures[i,:])
    return R

  def notifyFrameDone(self, frameIdx, landmark, mode):
    if mode == 'train' and np.random.rand() > 0.5:
      self.landmarkBoxes.append( {'box':self.groundTruth[self.imageList[frameIdx]][0], 'type':'gt_train'} )
      return
    if landmark is not None:
      self.landmarkBoxes.append( {'box':landmark, 'type':'fired'} )
      np.roll(self.landmarkFeatures,-1,axis=0)
      self.landmarkFeatures[0,:] = self.lastBoxFeatures
    elif len(self.landmarkBoxes) > 0:
      self.landmarkBoxes.append( {'box':self.landmarkBoxes[-1]['box'], 'type': 'copied'} )
    elif len(self.landmarkBoxes) == 0:
      self.landmarkBoxes.append( {'box':self.targetObjectBox, 'type':'target_copied'} )

  def getNextInitialBox(self):
    if len(self.landmarkBoxes) > 0:
      return self.landmarkBoxes[-1]
    else:
      return {'box':self.targetObjectBox, 'type':'target'}
      
  def saveRecord(self, frameIdx, record):
    record['gt'] = self.groundTruth[self.imageList[frameIdx]][0]
    self.boxLog.append(record)

  def dumpLog(self):
    if len(self.boxLog) > 0:
      with open(self.gtPath+'.log', 'w') as outfile:
        json.dump(self.boxLog, outfile)
    self.boxLog = []

