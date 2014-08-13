__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

#from pybrain.rl.learners.valuebased import ActionValueInterface
from pybrain.rl.learners.valuebased.interface import ActionValueInterface
from caffe import imagenet
import Image
import utils as cu
import numpy as np

class DeepQNetwork(ActionValueInterface):

  actions = 10
  batchSize = 50
  imageSize = 256
  contextPad = 16

  def __init__(self):
    self.net = None

  def loadNetwork(self, modelFile, networkFile, meanImage, cropSize, imgsDir):
    self.net = imagenet.ImageNetClassifier(modelFile, networkFile, IMAGE_DIM=imgDim, CROPPED_DIM=cropSize, MEAN_IMAGE=meanImage)
    self.net.caffenet.set_phase_test()
    self.net.caffenet.set_mode_gpu()
    self.imgsDir = imgsDir
    self.imgDim = imgDim
    self.cropSize = cropSize
    self.meanImage = meanImage
    
  def getMaxAction(self, state):
    activations = self.getActionValues(state)
    return np.argmax(activations, 1)

  def getActionValues(self, state):
    imgName = state[0]
    boxes = []
    for s in state[1]:
      boxes.append(s.nextBox)
    n = len(boxes)
    if self.net == None:
      return np.random.random([n, self.actions])

    activations = cu.emptyMatrix( [n, self.actions] )
    numBatches = (n + self.batchSize - 1) / self.batchSize
    boxes += [ [0,0,0,0] for x in range(numBatches * self.batchSize - n) ]

    dims = net.caffenet.InitializeImage(self.imgsDir + '/' + imgName, self.imgDim, self.meanImage, self.cropSize)
    for k in range(numBatches):
      s, f = k * self.batchSize, (k + 1) * self.batchSize
      e = self.batchSize if f <= n else n - s
      # Forward this batch
      net.caffenet.ForwardRegions(boxes[s:f], self.contextPad)
      outputs =  net.caffenet.blobs
      f = n if f > n else f
      # Collect outputs
      activations[s:f,:] = outputs['prob'].data[0:e,:,:,:].reshape([e,self.actions])
    # Release image data
    net.caffenet.ReleaseImageData()
    return activations
