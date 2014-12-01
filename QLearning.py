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
    totalMemorySize = memory.usableRecords
    replayMemorySize = config.geti('trainingIterationsPerBatch')*config.geti('trainingBatchSize')

    print '# Select a random sample of records'
    recordsToPull = [random.randint(0,totalMemorySize-2) for i in range(replayMemorySize)]
    samples = np.zeros( (replayMemorySize, memory.O.shape[1], 1, 1), np.float32 )
    targets = np.zeros( (replayMemorySize, 3, 1, 1), np.float32 )
    nextStates = np.zeros( (replayMemorySize, memory.O.shape[1], 1, 1), np.float32 )
    trainingSet = []
    terminalStates = []
    for i in range(len(recordsToPull)):
      r = recordsToPull[i]
      # Make sure that next state belongs to the same image
      if memory.I[r] != memory.I[r+1]:
        terminalStates.append(i) 
      samples[i,:,0,0] = memory.O[r,:]
      nextStates[i,:,0,0] = memory.O[r+1,:]
      action = memory.A[r,0]
      reward = memory.R[r,0]
      targets[i,:,0,0] = np.array([action, reward, 0.0], np.float32)
    if controller.net != None:
      controller.loadNetwork(definition='deploy.maxq.prototxt')
      discountedMaxNextQ = self.gamma*np.max( controller.getActivations(nextStates), axis=1 )
      discountedMaxNextQ[terminalStates] = 0.0
      targets[:,2,0,0] = discountedMaxNextQ

    print '# Update network'
    self.netManager.doNetworkTraining(samples, targets)

class CaffeMultiLayerPerceptronManagement():

  def __init__(self, workingDir):
    self.directory = workingDir
    self.writeSolverFile()
    self.solver = caffe.SGDSolver(self.directory + 'solver.prototxt')
    self.iter = 0
    self.itersPerEpisode = config.geti('trainingIterationsPerBatch')
    self.lr = config.getf('learningRate')
    self.stepSize = config.geti('stepSize')
    self.gamma = config.getf('gamma')
    print 'CAFFE SOLVER INITALIZED'

  def doNetworkTraining(self, samples, labels):
    self.solver.net.set_input_arrays(samples, labels)
    self.solver.solve()
    self.iter += config.geti('trainingIterationsPerBatch')
    if self.iter % self.stepSize == 0:
      newLR = self.lr * ( self.gamma** int(self.iter/self.stepSize) )
      print 'Changing LR to:',newLR
      self.solver.change_lr(newLR)

  def writeSolverFile(self):
    out = open(self.directory + '/solver.prototxt','w')
    out.write('train_net: "' + self.directory + 'train.prototxt"\n')
    out.write('base_lr: ' + config.get('learningRate') + '\n')
    out.write('lr_policy: "step"\n')
    out.write('gamma: ' + config.get('gamma') + '\n')
    out.write('stepsize: ' + config.get('stepSize') + '\n')
    out.write('display: 1\n')
    out.write('max_iter: ' + config.get('trainingIterationsPerBatch') + '\n')
    out.write('momentum: ' + config.get('momentum') + '\n')
    out.write('weight_decay: ' + config.get('weightDecay') + '\n')
    out.write('snapshot: ' + config.get('trainingIterationsPerBatch') + '\n')
    out.write('snapshot_prefix: "' + self.directory + 'multilayer_qlearner"\n')
    out.close()

