__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import json
from pybrain.rl.environments.environment import Environment
from pybrain.utilities import Named
import random

import BoxSearchState as bs
import learn.rl.RLConfig as config
import numpy as np
import utils.libDetection as det
import utils.utils as cu


def sigmoid(x, a=1.0, b=0.0):
  return 1.0/(1.0 + np.exp(-a*x + b))

def tanh(x, a=5, b=0.5, c=2.0):
  return c*np.tanh(a*x + b)

TEST_TIME_OUT = config.geti('testTimeOut')
ACTION_HISTORY_SIZE = config.geti('outputActions') * config.geti('actionHistoryLength')
MARK_WIDTH = config.getf('markWidth')

class BoxSearchEnvironment(Environment, Named):

  def __init__(self, imageList, mode, controller, groundTruthFile=None):
    self.mode = mode
    self.controller = controller
    self.testRecord = None
    self.idx = -1
    self.imageList = [x.strip() for x in open(imageList)]
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)
    #self.imageList = self.rankImages()
    #self.imageList = self.imageList[0:10]
    allImgs = set([x.strip() for x in open(config.get('allImagesList'))])
    self.negativeSamples = list(allImgs.difference(set(self.groundTruth.keys())))
    self.negativeEpisode = False
    if self.mode == 'train':
      self.negativeProbability = config.getf('negativeEpisodeProb')
      random.shuffle(self.imageList)
      #self.priorMemory = PriorMemory(config.get('allObjectsBoxes'), self.groundTruth, self.controller)
    self.loadNextEpisode()

  def performAction(self, action):
      self.state.performAction(action)

  def loadNextEpisode(self):
    self.episodeDone = False
    self.extraSteps = 5
    self.negativeEpisode = False
    if self.selectNegativeSample(): return
    # Save actions performed during this episode
    if self.mode == 'test' and self.testRecord != None:
      with open(config.get('testMemory') + self.imageList[self.idx] + '.txt', 'w') as outfile:
        json.dump(self.testRecord, outfile)
    # Load a new episode
    self.idx += 1
    if self.idx < len(self.imageList):
      # Initialize state
      if self.controller.net is not None:
        self.controller.net.prepareImage(self.imageList[self.idx])
      restartMode = {'train':'Random','test':'Full'}
      self.state = bs.BoxSearchState(self.imageList[self.idx], groundTruth=self.groundTruth, boxReset=restartMode[self.mode])
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
      if self.controller.net is not None:
        self.controller.net.prepareImage(self.negativeSamples[idx])
      self.state = bs.BoxSearchState(self.negativeSamples[idx], groundTruth=self.groundTruth, boxReset='Random')
      print 'Environment::LoadNextEpisode => Random Negative:',self.negativeSamples[idx]
      self.negativeEpisode = True

  def updatePostReward(self, reward, allDone, cover):
    if self.mode == 'test':
      self.testRecord['boxes'].append( self.state.box )
      self.testRecord['actions'].append( self.state.actionChosen )
      self.testRecord['values'].append( self.state.actionValue )
      self.testRecord['rewards'].append( reward )
      """
      self.testRecord['scores'].append( self.scores[:] )
      """
      if self.state.actionChosen == bs.PLACE_LANDMARK:
        if self.controller.net is not None:
          self.controller.net.coverRegion(self.state.box)
        self.state.reset()
      if self.state.stepsWithoutLandmark > TEST_TIME_OUT:
        self.state.reset('Quadrants')
    elif self.mode == 'train':
      # We do not cover false landmarks during training
      if self.state.actionChosen == bs.PLACE_LANDMARK and len(cover) > 0:
        # During training we only cover a carefully selected part of the ground truth box to avoid conflicts with other boxes.
        if self.controller.net is not None:
          self.controller.net.coverRegion(cover)
        self.state.reset('Random')
      if allDone:
        self.extraSteps -= 1
        if self.extraSteps <= 0:
          self.episodeDone = True

  def getSensors(self):
    """
    # Compute features of visible region (4096)
    activations = self.controller.getActivations(self.state.box)
    # Action history (90)
    actions = np.ones((ACTION_HISTORY_SIZE))*self.state.actionHistory

    # Concatenate all info in the state representation vector
    state = np.hstack( (activations[config.get('convnetLayer')], actions) )
    self.scores = activations['prob'][0:21].tolist()
    """
    return {'image':self.imageList[self.idx], 'state':self.state.box, 'negEpisode':self.negativeEpisode}

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

  def prepareImage(self, image):
    """ Copy an image to the GPU. Releases any previously loaded image. """
    if self.image != '':
      self.net.caffenet.ReleaseImageData()
    self.image = config.get('imageDir') + image + '.jpg'
    self.net.caffenet.InitializeImage(self.image, self.imgDim, self.imageMean, self.cropSize)

  def coverRegion(self, box, otherImg=None):
    """ Cover regions in an image copied to the GPU. The image to be covered is expected
    to be previously loaded. The given regions are defined by an array of boxes. """
    if otherImg is not None:
      boxes = [map(int,box)]
      self.net.caffenet.CoverRegions(boxes, config.get('imageDir') + otherImg + '.jpg', self.id)
    else:
      # Create two perpendicular boxes
      w = box[2]-box[0]
      h = box[3]-box[1]
      b1 = map(int, [box[0] + w*0.5 - w*MARK_WIDTH, box[1], box[0] + w*0.5 + w*MARK_WIDTH, box[3]])
      b2 = map(int, [box[0], box[1] + h*0.5 - h*MARK_WIDTH, box[2], box[1] + h*0.5 + h*MARK_WIDTH])
      boxes = [b1, b2]
      self.net.caffenet.CoverRegions(boxes, '', self.id)
    self.id += 1
    return True
