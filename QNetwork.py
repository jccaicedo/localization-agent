__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.learners.valuebased.interface import ActionValueInterface
import caffe
import os
import utils as cu
import numpy as np

import RLConfig as config

class QNetwork(ActionValueInterface):

  networkFile = config.get('networkDir') + config.get('snapshotPrefix') + '_iter_' + config.get('trainingIterationsPerBatch') + '.caffemodel'

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
    self.net = caffe.Net(modelFile, self.networkFile)
    self.net.set_phase_test()
    self.net.set_mode_gpu()
    
  def getMaxAction(self, state):
    values = self.getActionValues(state)
    return np.argmax(values, 1)

  def getActionValues(self, state):
    if self.net == None:
      return np.random.random([state.shape[0], 10])
    else:
      return self.getActivations(state)

  def getActivations(self, state):
    print self.net.inputs, state.shape
    out = self.net.forward_all( **{self.net.inputs[0]: state.reshape( (state.shape[0], state.shape[1], 1, 1) )} )
    return out['qvalues'].squeeze(axis=(2,3))

