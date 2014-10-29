__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import random
import Image
import CaffeMultiLayerPerceptronManagement as cnm
import SingleObjectLocalizer as sol
import RLConfig as config
import numpy as np

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class QLearning(ValueBasedLearner):

  offPolicy = True
  batchMode = True
  dataset = []

  trainingSamples = 0

  def __init__(self, alpha=0.5, gamma=0.99):
    ValueBasedLearner.__init__(self)
    self.alpha = alpha
    self.gamma = gamma
    self.netManager = cnm.CaffeMultiLayerPerceptronManagement(config.get('networkDir'))

  def learn(self, data, controller):
    images = []
    hash = {}
    print 'MEMORY SIZE:',len(data)
    for d in data:
      self.dataset.append(d)
    self.updateTrainingDatabase(controller)
    self.netManager.doNetworkTraining()

  def updateTrainingDatabase(self, controller):
    trainRecs, numTrain = self.netManager.readTrainingDatabase('training.txt')
    trainRecs = self.dropRecords(trainRecs, numTrain, len(self.dataset))
    valRecs, numVal = self.netManager.readTrainingDatabase('validation.txt')
    valRecs = self.dropRecords(valRecs, numVal, len(self.dataset))
    trainRecs, valRecs = self.mergeDatasetAndRecords(trainRecs, valRecs)
    trainRecs = self.computeNextMaxQ(controller, trainRecs)
    #valRecs = self.computeNextMaxQ(controller, valRecs)
    self.trainingSamples = self.netManager.saveDatabaseFile(trainRecs, 'training.txt')
    self.netManager.saveDatabaseFile(valRecs, 'validation.txt')
    self.dataset = []
    
  def dropRecords(self, rec, total, new):
    random.shuffle(rec)
    end = min( config.geti('replayMemorySize') - new, total )
    rec = rec[0:end]
    return rec

  def mergeDatasetAndRecords(self, train, val):
    numTrain = len(self.dataset)*(1 - config.getf('percentOfValidation'))
    numVal = len(self.dataset)*config.getf('percentOfValidation')
    random.shuffle( self.dataset )
    for i in range(len(self.dataset)):
      # record format: Action, reward, discountedMaxQ, all_state_features
      record = [ self.dataset[i]['A'], self.dataset[i]['R'], 0.0 ] + self.dataset[i]['O'].tolist()

      if i < numTrain:
        train.append(record)
      else:
        val.append(record)

    return train, val

  def computeNextMaxQ(self, controller, records):
    print 'Computing discounted reward for all memory samples'
    if controller.net == None:
      return records
    return records
