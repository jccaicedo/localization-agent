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

  def learn(self, controller):
    print '# Identify memory records stored by the agent'
    recordSize = config.geti('trainInteractions')
    memoryIndex = [f for f in os.listdir(config.get('trainMemory')) if f.endswith('.mat')]
    memoryIndex.sort()
    totalMemorySize = recordSize*len(memoryIndex)
    replayMemorySize = config.geti('trainingIterationsPerBatch')*config.geti('trainingBatchSize')

    print '# Select a random sample of records'
    recordsToPull = [random.randint(0,totalMemorySize-1) for i in range(replayMemorySize)]
    imageKey = lambda x: x/recordSize
    recordKey = lambda x: int(recordSize*(x/float(recordSize) - x/recordSize))
    recordsToPull = [ [imageKey(r),recordKey(r)] for r in recordsToPull]
    records = {}
    for r in recordsToPull:
      try: records[r[0]].append( r[1] )
      except: records[r[0]] = [ r[1] ]

    print '# Read samples to build the training batch'
    trainingSamples = []
    for img in records.keys():
      data = scipy.io.loadmat(config.get('trainMemory') + memoryIndex[img])
      for key in records[img]:
        if key == data['time'].shape[0]-1:
          # Shift backward in time to see features in next state
          key = data['time'].shape[0]-2 
        state0 = data['observations'][key,:]
        action = data['actions'][key,0]
        reward = data['rewards'][key,0]
        state1 = data['observations'][key+1,:]
        trainingSamples.append( {'S0':state0, 'A':action, 'R':reward, 'S1':state1} )
    
    print '# Compute MaxNextQ for training records'
    if controller.net == None:
      maxNextQ = lambda x: 0.0
    else:
      maxNextQ = lambda x: np.max( controller.getActivations(x) )
    trainingSet = []
    for sample in trainingSamples:
      discountedMaxNextQ = self.gamma*maxNextQ( sample['S1'].reshape((1,sample['S1'].shape[0],1,1)) )
      trainingSet.append( [sample['A'], sample['R'], discountedMaxNextQ] + sample['S0'].tolist() )
    
    print '# Save training set and start network update'
    self.netManager.saveDatabaseFile(trainingSet, 'training.txt')
    self.netManager.saveDatabaseFile(trainingSet[0:10], 'validation.txt')
    self.netManager.doNetworkTraining()

