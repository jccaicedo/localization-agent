__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import random
import numpy as np
import scipy.io
import caffe

import RLConfig as config
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class QLearning(ValueBasedLearner):

  offPolicy = True
  batchMode = True
  dataset = []

  trainingSamples = 0

  def __init__(self, alpha=0.5):
    ValueBasedLearner.__init__(self)
    self.alpha = alpha
    self.gamma = config.getf('gammaDiscountReward')
    self.netManager = CaffeMultiLayerPerceptronManagement(config.get('networkDir'))

  def learn(self, memory, controller):
    print '# Identify memory records stored by the agent',memory.O.shape, memory.A.shape,memory.R.shape
    recordingSize = config.geti('trainInteractions')
    totalMemorySize = memory.O.shape[0]
    replayMemorySize = config.geti('trainingIterationsPerBatch')*config.geti('trainingBatchSize')

    if controller.net == None:
      maxNextQ = lambda x: 0.0
    else:
      maxNextQ = lambda x: np.max( controller.getActivations(x) )

    print '# Select a random sample of records'
    recordsToPull = [random.randint(0,totalMemorySize-1) for i in range(replayMemorySize)]
    samples = np.zeros( (replayMemorySize, memory.O.shape[1], 1, 1), np.float32 )
    targets = np.zeros( (replayMemorySize, 3, 1, 1), np.float32 )
    trainingSet = []
    for i in range(len(recordsToPull)):
      r = recordsToPull[i]
      if (r+1) % recordingSize == 0:
        # Shift backward in time to see features in next state
        r = r - 1 
      samples[i,:,0,0] = memory.O[r,:]
      action = memory.A[r,0]
      reward = memory.R[r,0]
      nextState = memory.O[r+1,:]
      discountedMaxNextQ = self.gamma*maxNextQ( nextState.reshape((1,nextState.shape[0],1,1)) )
      targets[i,:,0,0] = np.array([action, reward, discountedMaxNextQ], np.float32)

    print '# Update network'
    self.netManager.doNetworkTraining(samples, targets)

class CaffeMultiLayerPerceptronManagement():

  def __init__(self, workingDir):
    self.directory = workingDir
    self.writeSolverFile()
    self.solver = caffe.SGDSolver(self.directory + 'solver.prototxt')
    print 'CAFFE SOLVER INITALIZED'

  def doNetworkTraining(self, samples, labels):
    self.solver.net.set_input_arrays(samples, labels)
    self.solver.solve()

  def writeSolverFile(self):
    out = open(self.directory + '/solver.prototxt','w')
    out.write('train_net: "' + self.directory + 'train.prototxt"\n')
    out.write('base_lr: ' + config.get('learningRate') + '\n')
    out.write('lr_policy: "step"\n')
    out.write('gamma: ' + config.get('gamma') + '\n')
    out.write('stepsize: 10000\n')
    out.write('display: 1\n')
    out.write('max_iter: ' + config.get('trainingIterationsPerBatch') + '\n')
    out.write('momentum: ' + config.get('momentum') + '\n')
    out.write('weight_decay: ' + config.get('weightDecay') + '\n')
    out.write('snapshot: ' + config.get('trainingIterationsPerBatch') + '\n')
    out.write('snapshot_prefix: "' + self.directory + 'multilayer_qlearner"\n')
    out.close()

