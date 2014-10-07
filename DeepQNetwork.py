__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

#from pybrain.rl.learners.valuebased import ActionValueInterface
from pybrain.rl.learners.valuebased.interface import ActionValueInterface
from caffe import imagenet
import Image
import os
import utils as cu
import numpy as np

import RLConfig as config

class DeepQNetwork(ActionValueInterface):

  networkFile = config.get('networkDir') + config.get('snapshotPrefix') + '_iter_' + config.get('trainingIterationsPerBatch')

  def __init__(self):
    self.net = None
    print self.networkFile
    if os.path.exists(self.networkFile):
      self.loadNetwork()

  def releaseNetwork(self):
    if self.net != None:
      del self.net
      self.net = None

  def loadNetwork(self):
    modelFile = config.get('networkDir') + 'deploy.prototxt'
    meanImage = config.get('meanImagePickle')
    self.net = imagenet.ImageNetClassifier(modelFile, self.networkFile, IMAGE_DIM=config.geti('imageSize'), CROPPED_DIM=config.geti('cropSize'), MEAN_IMAGE=meanImage)
    self.net.caffenet.set_phase_test()
    self.net.caffenet.set_mode_gpu()
    self.meanImage = self.net._IMAGENET_MEAN.swapaxes(1, 2).swapaxes(0, 1).astype('float32')
    
  def getMaxAction(self, state):
    values = self.getActionValues(state)
    return np.argmax(values, 1)

  def getActionValues(self, state):
    imgName = state[0]
    if self.net == None:
      return np.random.random([len(state[1]), config.geti('outputActions')])
    else:
      boxes = []
      stateFeatures = np.zeros( (len(state[1]), 20, 1, 1), dtype=np.float32)
      for i in range(len(state[1])):
        s = state[1][i]
        boxes.append( map(int, s.nextBox) )
        stateFeatures[i,0,0,0] = s.prevScore
        stateFeatures[i,1:5,0,0] = np.asarray( s.normPrevBox() )
        stateFeatures[i,5,0,0] = s.currScore
        stateFeatures[i,6:10,0,0] = np.asarray( s.normCurrBox() )
        if s.prevAction() >= 0:
          stateFeatures[i,10 + s.prevAction(), 0,0] = 1.0
      return self.getActivations(config.get('imageDir') + '/' + imgName + '.jpg', boxes, stateFeatures)

  def getActivations(self, imagePath, boxes, state):
    n = len(boxes)
    activations = cu.emptyMatrix( [n, config.geti('outputActions')] )
    numBatches = (n + config.geti('deployBatchSize') - 1) / config.geti('deployBatchSize')
    boxes += [ [0,0,0,0] for x in range(numBatches * config.geti('deployBatchSize') - n) ]
    stateFeatures = np.zeros( (len(boxes), 20, 1, 1), dtype=np.float32)
    stateFeatures[0:n,:,:,:] = state

    dims = self.net.caffenet.InitializeImage(imagePath, config.geti('imageSize'), self.meanImage, config.geti('cropSize'))
    for k in range(numBatches):
      s, f = k * config.geti('deployBatchSize'), (k + 1) * config.geti('deployBatchSize')
      e = config.geti('deployBatchSize') if f <= n else n - s
      # Forward this batch
      #self.net.caffenet.ForwardRegions(boxes[s:f], config.geti('contextPad'))
      self.net.caffenet.ForwardRegionsAndState(boxes[s:f], config.geti('contextPad'), [stateFeatures[s:f,:,:,:]])
      outputs =  self.net.caffenet.blobs
      f = n if f > n else f
      # Collect outputs
      activations[s:f,:] = outputs['prob'].data[0:e,:,:,:].reshape([e,config.geti('outputActions')])
    # Release image data
    self.net.caffenet.ReleaseImageData()
    return activations

