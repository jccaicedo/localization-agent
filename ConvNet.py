__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import utils as cu
import numpy as np

import caffe
from caffe import wrapperv0

import RLConfig as config

LAYER = config.get('convnetLayer')

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
    outputs =  self.net.caffenet.blobs
    result = {'prob':outputs['prob'].data.squeeze(), LAYER:outputs[LAYER].data.squeeze()}
    return result

  def coverRegion(self, box):
    boxes = [map(int,box)]
    self.net.caffenet.CoverRegions(boxes, self.id)
    self.id += 1
    return True

