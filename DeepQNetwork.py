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
    #terminalScores = values[:,0:2]
    return np.argmax(values, 1)

  def getActionValues(self, state):
    imgName = state[0]
    boxes = []
    for s in state[1]:
      boxes.append( map(int, s.nextBox) )
    if self.net == None:
      return np.random.random([len(boxes), config.geti('outputActions')])
    else:
      return self.getActivations(config.get('imageDir') + '/' + imgName + '.jpg', boxes)

  def getActivations(self, imagePath, boxes):
    n = len(boxes)
    activations = cu.emptyMatrix( [n, config.geti('outputActions')] )
    numBatches = (n + config.geti('deployBatchSize') - 1) / config.geti('deployBatchSize')
    boxes += [ [0,0,0,0] for x in range(numBatches * config.geti('deployBatchSize') - n) ]

    dims = self.net.caffenet.InitializeImage(imagePath, config.geti('imageSize'), self.meanImage, config.geti('cropSize'))
    for k in range(numBatches):
      s, f = k * config.geti('deployBatchSize'), (k + 1) * config.geti('deployBatchSize')
      e = config.geti('deployBatchSize') if f <= n else n - s
      # Forward this batch
      self.net.caffenet.ForwardRegions(boxes[s:f], config.geti('contextPad'))
      outputs =  self.net.caffenet.blobs
      f = n if f > n else f
      # Collect outputs
      activations[s:f,:] = outputs['prob'].data[0:e,:,:,:].reshape([e,config.geti('outputActions')])
    # Release image data
    self.net.caffenet.ReleaseImageData()
    return activations

