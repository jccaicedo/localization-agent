__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

#from pybrain.rl.learners.valuebased import ActionValueInterface
from pybrain.rl.learners.valuebased.interface import ActionValueInterface
#from caffe import imagenet
#import Image
import os
import utils as cu
import numpy as np

#import RLConfig as config

class QNetwork(ActionValueInterface):

  #networkFile = config.networkDir + config.SNAPSHOOT_PREFIX + '_iter_' + str(config.trainingIterationsPerBatch)

  def __init__(self):
    self.net = None
    '''print self.networkFile
    if os.path.exists(self.networkFile):
      self.loadNetwork()'''

  def releaseNetwork(self):
    if self.net != None:
      del self.net
      self.net = None

  def loadNetwork(self):
    return
    '''modelFile = config.networkDir + 'deploy.prototxt'
    meanImage = config.MEAN_IMAGE_PICKLE
    self.net = imagenet.ImageNetClassifier(modelFile, self.networkFile, IMAGE_DIM=config.imageSize, CROPPED_DIM=config.cropSize, MEAN_IMAGE=meanImage)
    self.net.caffenet.set_phase_test()
    self.net.caffenet.set_mode_gpu()
    self.meanImage = self.net._IMAGENET_MEAN.swapaxes(1, 2).swapaxes(0, 1).astype('float32')'''
    
  def getMaxAction(self, state):
    values = self.getActionValues(state)
    return np.argmax(values, 1)

  def getActionValues(self, state):
    #imgName = state[0]
    #boxes = []
    #for s in state[1]:
    #  boxes.append( map(int, s.nextBox) )
    if self.net == None:
      return np.random.random([state.shape[0], 10])
    else:
      pass
      #return self.getActivations(config.imageDir + '/' + imgName + '.jpg', boxes)

  def getActivations(self, imagePath, boxes):
    return None
    '''n = len(boxes)
    activations = cu.emptyMatrix( [n, config.outputActions] )
    numBatches = (n + config.deployBatchSize - 1) / config.deployBatchSize
    boxes += [ [0,0,0,0] for x in range(numBatches * config.deployBatchSize - n) ]

    dims = self.net.caffenet.InitializeImage(imagePath, config.imageSize, self.meanImage, config.cropSize)
    for k in range(numBatches):
      s, f = k * config.deployBatchSize, (k + 1) * config.deployBatchSize
      e = config.deployBatchSize if f <= n else n - s
      # Forward this batch
      self.net.caffenet.ForwardRegions(boxes[s:f], config.contextPad)
      outputs =  self.net.caffenet.blobs
      f = n if f > n else f
      # Collect outputs
      activations[s:f,:] = outputs['prob'].data[0:e,:,:,:].reshape([e,config.outputActions])
    # Release image data
    self.net.caffenet.ReleaseImageData()
    return activations'''


