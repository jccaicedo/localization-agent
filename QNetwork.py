__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.learners.valuebased.interface import ActionValueInterface
import caffe
import os
import utils as cu
import numpy as np

import RLConfig as config

class QNetwork(ActionValueInterface):

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
    n = state.shape[0]
    activations = cu.emptyMatrix( [n, config.geti('outputActions')] )
    numBatches = (n + config.geti('deployBatchSize') - 1) / config.geti('deployBatchSize')

    if n >= config.geti('deployBatchSize'):
      for k in range(numBatches):
        s, f = k * config.geti('deployBatchSize'), (k + 1) * config.geti('deployBatchSize')
        e = config.geti('deployBatchSize') if f <= n else n - s
        # Forward this batch
        out = [np.zeros((config.geti('deployBatchSize'), config.geti('outputActions'), 1, 1), dtype=np.float32)]
        inp = [ np.zeros( [config.geti('deployBatchSize'), state.shape[1], 1, 1], dtype=np.float32 ) ]
        inp[0][:,:,0,0] = state[s:f]
        self.net.Forward( inp, out )
        outputs =  self.net.blobs
        f = n if f > n else f
        # Collect outputs
        activations[s:f,:] = outputs['prob'].data[0:e,:,:,:].reshape([e,config.geti('outputActions')])
    else:
      # Forward this batch
      out = [np.zeros((config.geti('deployBatchSize'), config.geti('outputActions'), 1, 1), dtype=np.float32)]
      inp = [ np.zeros( [config.geti('deployBatchSize'), state.shape[1], 1, 1], dtype=np.float32 ) ]
      inp[0][0:n,:,0,0] = state[0:n]
      self.net.Forward( inp, out )
      outputs =  self.net.blobs
      # Collect outputs
      activations[0:n,:] = outputs['prob'].data[0:n,:,:,:].reshape([n,config.geti('outputActions')])
     
    return activations

