__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
from pybrain.rl.learners.valuebased.interface import ActionValueInterface
import random

import caffe
import learn.rl.RLConfig as config
import numpy as np


EXPLORE = 0
EXPLOIT = 1

def defaultSampler():
  return np.random.random([1, config.geti('outputActions')])

class QNetwork(ActionValueInterface):

  networkFile = config.get('networkDir') + config.get('snapshotPrefix') + '_iter_' + config.get('trainingIterationsPerBatch') + '.caffemodel'

  def __init__(self):
    self.net = None
    print 'QNetwork::Init. Loading ',self.networkFile
    self.loadNetwork()
    self.sampler = defaultSampler

  def releaseNetwork(self):
    if self.net != None:
      del self.net
      self.net = None

  def loadNetwork(self, definition='deploy.prototxt'):
    if os.path.isfile(self.networkFile):
      self.contextPad = config.geti('contextPad')
      modelFile = config.get('networkDir') + definition
      self.net = caffe.Net(modelFile, self.networkFile)
      self.net.set_phase_test()
      self.net.set_mode_gpu()
      print 'QNetwork loaded'
    else:
      self.net = None
      print 'QNetwork not found'
    
  def getMaxAction(self, state):
    values = self.getActionValues(state)
    return np.argmax(values, 1)

  def getActionValues(self, state):
    if self.net == None or self.exploreOrExploit() == EXPLORE:
      return self.sampler()
    else:
      return self.getActivations(state)

  def getActivations(self, state):
    """ Obtains the action values for the QNetwork by forwarding the regions
    which are here represented by a given array of boxes with integer coordinates.
    """
    
    """
    out = self.net.forward_all( **{self.net.inputs[0]: boxes.reshape( (boxes.shape[0], boxes.shape[1], 1, 1) )} )
    """
    
    boxes = [map(int, self.state.box)]
    self.net.caffenet.ForwardRegions(boxes, self.contextPad)
    out = self.net.caffenet.blobs
    
    return out['qvalues'].squeeze(axis=(2,3))

  def setEpsilonGreedy(self, epsilon, sampler=None):
    if sampler is not None:
      self.sampler = sampler
    self.epsilon = epsilon

  def exploreOrExploit(self):
    if self.epsilon > 0:
      if random.random() < self.epsilon:
        return EXPLORE
    return EXPLOIT
