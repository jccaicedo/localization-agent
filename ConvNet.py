__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import utils as cu
import numpy as np

import caffe
from caffe import wrapperv0

import RLConfig as config

LAYER = config.get('convnetLayer')
MARK_WITH = config.getf('markWidth')

class ConvNet():

  def __init__(self):
    self.net = None
    self.image = ''
    self.id = 0
    self.loadNetwork()

  def loadNetwork(self):
    self.imgDim = config.geti('imageDim')
    self.cropSize = config.geti('cropSize')
    self.contextPad = config.geti('contextPad')
    #self.stateContextFactor = config.geti('stateContextFactor')
    modelFile = config.get('convnetDir') + config.get('convNetDef')
    networkFile = config.get('convnetDir') + config.get('trainedConvNet')
    self.net = wrapperv0.ImageNetClassifier(modelFile, networkFile, IMAGE_DIM=self.imgDim, CROPPED_DIM=self.cropSize, MEAN_IMAGE=config.get('meanImage'))
    self.net.caffenet.set_mode_gpu()
    self.net.caffenet.set_phase_test()
    self.imageMean = self.net._IMAGENET_MEAN.swapaxes(1, 2).swapaxes(0, 1).astype('float32')

  def prepareImage(self, image):
    if self.image != '':
      self.net.caffenet.ReleaseImageData()
    self.image = config.get('imageDir') + image + '.jpg'
    self.net.caffenet.InitializeImage(self.image, self.imgDim, self.imageMean, self.cropSize)

  def getActivations(self, box):
    boxes = [map(int,box)]
    self.net.caffenet.ForwardRegions(boxes, self.contextPad)
    outputsImg =  self.net.caffenet.blobs
    #if self.stateContextFactor == 5:
    #  self.net.caffenet.ForwardRegionsOnSource(boxes, self.contextPad*self.stateContextFactor, 1)
    #else:
    #  # Adaptive context pad
    #  w = box[2] - box[0]
    #  h = box[3] - box[1]
    #  if w > h:
    #    c = h/2
    #  else:
    #    c = w/2
    #  self.net.caffenet.ForwardRegionsOnSource(boxes, int(c), 1)
    #outputsStt =  self.net.caffenet.blobs
    result = {'prob':outputsImg['prob'].data.squeeze(), LAYER:outputsImg[LAYER].data.squeeze()}
    #result = {'prob':outputsImg['prob'].data.squeeze(), LAYER:outputsImg[LAYER].data.squeeze(), LAYER+'_stt':outputsStt[LAYER].data.squeeze()}
    return result

  def coverRegion(self, box, otherImg=None):
    if otherImg is not None:
      boxes = [map(int,box)]
      self.net.caffenet.CoverRegions(boxes, config.get('imageDir') + otherImg + '.jpg', self.id)
    else:
      # Create two perpendicular boxes
      w = box[2]-box[0]
      h = box[3]-box[1]
      b1 = map(int, [box[0] + w*0.5 - w*MARK_WITH, box[1], box[0] + w*0.5 + w*MARK_WITH, box[3]])
      b2 = map(int, [box[0], box[1] + h*0.5 - h*MARK_WITH, box[2], box[1] + h*0.5 + h*MARK_WITH])
      boxes = [b1, b2]
      self.net.caffenet.CoverRegions(boxes, '', self.id)
    self.id += 1
    return True

