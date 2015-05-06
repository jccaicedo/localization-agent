__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.learners.valuebased.interface import ActionValueInterface
import caffe
import os
import utils as cu
import numpy as np
import random

import RLConfig as config

EXPLORE = 0
EXPLOIT = 1

def defaultSampler():
  return np.random.random([1, config.geti('outputActions')])

class QNetwork(ActionValueInterface):

  networkFile = config.get('networkDir') + config.get('snapshotPrefix') + '_iter_' + config.get('trainingIterationsPerBatch') + '.caffemodel'

  def __init__(self):
    self.net = None
    print 'QNetwork::Init. Loading ',self.networkFile
    if os.path.exists(self.networkFile):
      self.loadNetwork()
    self.sampler = defaultSampler

  def releaseNetwork(self):
    if self.net != None:
      del self.net
      self.net = None

  def loadNetwork(self, definition='deploy.prototxt'):
    if os.path.isfile(self.networkFile):
      modelFile = config.get('networkDir') + definition
      self.net = caffe.Net(modelFile, self.networkFile)
      self.net.set_phase_test()
      self.net.set_mode_gpu()
    else:
      self.net = None
    
  def getMaxAction(self, state):
    values = self.getActionValues(state)
    return np.argmax(values, 1)

  def getActionValues(self, state):
    if self.net == None or self.exploreOrExploit() == EXPLORE:
      return self.sampler()
    else:
      return self.getActivations(state)

  def getActivations(self, state):
    out = self.net.forward_all( **{self.net.inputs[0]: state.reshape( (state.shape[0], state.shape[1], 1, 1) )} )
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
