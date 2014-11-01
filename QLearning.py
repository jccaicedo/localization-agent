__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import random
import Image
import CaffeMultiLayerPerceptronManagement as cnm
import SingleObjectLocalizer as sol
import RLConfig as config
import numpy as np
import scipy.io

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
    self.netManager = cnm.CaffeMultiLayerPerceptronManagement(config.get('networkDir'))

  def learn(self, memory, controller):
    print '# Identify memory records stored by the agent',memory.O.shape, memory.A.shape,memory.R.shape
    recordSize = config.geti('trainInteractions')
    totalMemorySize = memory.O.shape[0]
    replayMemorySize = config.geti('trainingIterationsPerBatch')*config.geti('trainingBatchSize')

    if controller.net == None:
      maxNextQ = lambda x: 0.0
    else:
      maxNextQ = lambda x: np.max( controller.getActivations(x) )

    print '# Select a random sample of records'
    recordsToPull = [random.randint(0,totalMemorySize-1) for i in range(replayMemorySize)]
    trainingSet = []
    for r in recordsToPull:
      if (r+1) % recordSize == 0:
        # Shift backward in time to see features in next state
        r = r - 1 
      state0 = memory.O[r,:]
      action = memory.A[r,0]
      reward = memory.R[r,0]
      state1 = memory.O[r+1,:]
      discountedMaxNextQ = self.gamma*maxNextQ( state1.reshape((1,state1.shape[0],1,1)) )
      trainingSet.append( [action, reward, discountedMaxNextQ] + state0.tolist() )

    print '# Save training set and start network update'
    self.netManager.saveDatabaseFile(trainingSet, 'training.txt')
    self.netManager.saveDatabaseFile(trainingSet[0:10], 'validation.txt')
    self.netManager.doNetworkTraining()

