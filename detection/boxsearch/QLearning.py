__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
import random

import BoxSearchState as bs
import caffe
import learn.rl.RLConfig as config
import numpy as np


DETECTION_REWARD = config.getf('detectionReward')
ACTION_HISTORY_SIZE = bs.NUM_ACTIONS*config.geti('actionHistoryLength')
ACTION_HISTORY_LENTH = config.geti('actionHistoryLength')
NETWORK_INPUTS = config.geti('stateFeatures')/config.geti('temporalWindow')
REPLAY_MEMORY_SIZE = config.geti('trainingIterationsPerBatch')*config.geti('trainingBatchSize')

def generateRandomActionHistory():
  actions = np.zeros((ACTION_HISTORY_SIZE))
  history = [i*bs.NUM_ACTIONS + np.random.randint(0,bs.PLACE_LANDMARK) for i in range(ACTION_HISTORY_LENTH)]
  actions[history] = 1
  return actions

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

    print '# Select a random sample of records'
    recordsToPull = [random.randint(0,totalMemorySize-2) for i in range(REPLAY_MEMORY_SIZE)]
    samples = np.zeros( (REPLAY_MEMORY_SIZE, memory.O.shape[1], 1, 1), np.float32 )
    targets = np.zeros( (REPLAY_MEMORY_SIZE, 3, 1, 1), np.float32 )
    nextStates = np.zeros( (REPLAY_MEMORY_SIZE, memory.O.shape[1], 1, 1), np.float32 )
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

  def learnFromPriors(self, priors):
    print '# Prior records given to the agent', priors.N.shape, priors.P.shape
    negativeRecords = priors.N.shape[0]
    maxNegAllowed = REPLAY_MEMORY_SIZE - priors.P.shape[0]

    print '# Select a random sample of records'
    recordsToPull = [random.randint(0,negativeRecords-1) for i in range(maxNegAllowed)]
    samples = np.zeros( (REPLAY_MEMORY_SIZE, NETWORK_INPUTS, 1, 1), np.float32 )
    targets = np.zeros( (REPLAY_MEMORY_SIZE, 3, 1, 1), np.float32 )
    trainingSet = []
    for i in range(len(recordsToPull)):
      r = recordsToPull[i]
      samples[i,:,0,0] = np.hstack( (priors.N[r,:], generateRandomActionHistory()) )
      targets[i,:,0,0] = np.array([bs.PLACE_LANDMARK, -DETECTION_REWARD, 0.0], np.float32)
    j = 0
    for i in range(len(recordsToPull), REPLAY_MEMORY_SIZE):
      samples[i,:,0,0] = np.hstack( (priors.P[j,:], generateRandomActionHistory()) )
      targets[i,:,0,0] = np.array([bs.PLACE_LANDMARK, DETECTION_REWARD, 0.0], np.float32)
      j += 1

    print '# Update network'
    np.random.shuffle(samples)
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

