__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import random
import Image
#import CaffeNetworkManagement as cnm
import SingleObjectLocalizer as sol
import RLConfig as config
import numpy as np

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class DeepQLearning(ValueBasedLearner):

  offPolicy = True
  batchMode = True
  dataset = []

  trainingSamples = 0

  def __init__(self, alpha=0.5, gamma=0.99):
    ValueBasedLearner.__init__(self)
    self.alpha = alpha
    self.gamma = gamma
    #self.netManager = cnm.CaffeNetworkManagement(config.networkDir)

  def learn(self, data, controller):
    images = []
    hash = {}
    for d in data:
      images.append(d[0])
      key = '_'.join(map(str, d[0:-2]))
      try:
        exists = hash[key]
      except:
        self.dataset.append(d)
        hash[key] = True
        print 'State',d
    #self.updateTrainingDatabase(controller)
    #self.netManager.doNetworkTraining()

  def updateTrainingDatabase(self, controller):
    trainRecs, numTrain = self.netManager.readTrainingDatabase('training.txt')
    trainRecs = self.dropRecords(trainRecs, numTrain, len(self.dataset))
    valRecs, numVal = self.netManager.readTrainingDatabase('validation.txt')
    valRecs = self.dropRecords(valRecs, numVal, len(self.dataset))
    trainRecs, valRecs = self.mergeDatasetAndRecords(trainRecs, valRecs)
    trainRecs = self.computeNextMaxQ(controller, trainRecs)
    valRecs = self.computeNextMaxQ(controller, valRecs)
    self.trainingSamples = self.netManager.saveDatabaseFile(trainRecs, 'training.txt')
    self.netManager.saveDatabaseFile(valRecs, 'validation.txt')
    self.dataset = []
    
  def dropRecords(self, rec, total, new):
    if total > config.replayMemorySize:
      drop = 0
      while drop < new:
        for k in rec.keys():
          rec[k].pop(0)
          drop += 1
    return rec

  def mergeDatasetAndRecords(self, train, val):
    numTrain = len(self.dataset)*(1 - config.percentOfValidation)
    numVal = len(self.dataset)*config.percentOfValidation
    random.shuffle( self.dataset )
    for i in range(len(self.dataset)):
      imgPath = config.imageDir + self.dataset[i][0] + '.jpg'
      # record format: Action, reward, discountedMaxQ, x1, y1, x2, y2,
      record = [self.dataset[i][10], self.dataset[i][11], 0.0] + self.dataset[i][1:5]

      if i < numTrain:
        try: 
          train[imgPath].append(record)
        except: 
          train[imgPath] = [ record ]
      else:
        try: val[imgPath].append(record)
        except: 
          val[imgPath] = [ record ]

    return train, val

  def computeNextMaxQ(self, controller, records):
    print 'Computing discounted reward for all memory samples'
    if controller.net == None:
      return records
    for img in records.keys():
      imSize = Image.open(img).size
      boxes = []
      for i in range(len(records[img])):
        ol = sol.SingleObjectLocalizer(imSize, records[img][i][3:])
        ol.performAction(records[img][i][0])
        boxes.append( map(int, ol.nextBox) )
      maxQ = np.max( controller.getActivations(img, boxes), 1 )
      for i in range(len(maxQ)):
        if records[img][i][0] > 1: # Not a terminal action
          records[img][i][2] = self.gamma*maxQ[i]
        else:
          records[img][i][2] = 0.0
    return records
