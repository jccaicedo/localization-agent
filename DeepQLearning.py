__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import os
import random
import Image

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class DeepQLearning(ValueBasedLearner):

  maxRecordsInDatabase = 2000000
  percentOfValidation = 0.1

  imageDir = '/u/sciteam/caicedor/scratch/pascalImgs/'
  workingDir = '/u/sciteam/caicedor/scratch/rldetector01/'
  trainingFile = 'training.txt'
  validationFile = 'validation.txt'

  offPolicy = True
  batchMode = True
  dataset = []

  trainingSamples = 0

  def __init__(self, alpha=0.5, gamma=0.99):
    ValueBasedLearner.__init__(self)
    self.alpha = alpha
    self.gamma = gamma

  def learn(self, data):
    images = []
    hash = {}
    for d in data:
      images.append(d[0])
      key = '_'.join(map(str, d[0:-2]))
      try:
        exists = hash[key]
        #print '*',d
      except:
        self.dataset.append(d)
        hash[key] = True
        #print d
    #print len(data),len(set(images))
    self.updateTrainingDatabase()
    self.startFinetuning()
    
  def startFinetuning(self):
    # If the network already exists, keep training it?
    if self.trainingSamples == 0:
      return
    elif self.trainingSamples > 0 and self.trainingSamples < 100000:
      # finetune with 
      pass

  def updateTrainingDatabase(self):
    trainRecs, numTrain = self.readTrainingDatabase(self.trainingFile)
    trainRecs = self.dropRecords(trainRecs, numTrain, len(self.dataset))
    valRecs, numVal = self.readTrainingDatabase(self.validationFile)
    valRecs = self.dropRecords(valRecs, numVal, len(self.dataset))
    trainRecs, valRecs = self.mergeDatasetAndRecords(trainRecs, valRecs)
    self.trainingSamples = self.saveDatabaseFile(trainRecs, self.trainingFile)
    self.saveDatabaseFile(valRecs, self.validationFile)
    self.dataset = []
    
  def readTrainingDatabase(self, file):
    records = {}
    total = 0
    if os.path.isfile(self.workingDir + file):
      data = [x.split() for x in open(self.workingDir + file)]
      img = ''
      for i in range(len(data)):
        if data[i][0].startswith('#'):
          img = data[i+1][0]
          meta = [ data[i+3][0], data[i+4][0] ]
          i = i + 6
          records[img] = []
        elif len(data[i]) > 2:
          records[img].append( map(float, data[i]) )
          records[img][-1][0] -= 1.0
      total = len(data) - 6*len(records)
    return records, total

  def dropRecords(self, rec, total, new):
    if total > self.maxRecordsInDatabase:
      drop = 0
      while drop < new:
        for k in rec.keys():
          rec[k].pop(0)
          drop += 1
    return rec

  def mergeDatasetAndRecords(self, train, val):
    numTrain = len(self.dataset)*(1 - self.percentOfValidation)
    numVal = len(self.dataset)*self.percentOfValidation
    random.shuffle( self.dataset )
    for i in range(len(self.dataset)):
      imgPath = self.imageDir + self.dataset[i][0] + '.jpg'
      # record format: Action, reward, discountedMaxQ, x1, y1, x2, y2,
      record = [self.dataset[i][10], self.dataset[i][11], 0.5] + self.dataset[i][1:5]

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

  def saveDatabaseFile(self, records, outputFile):
    out = open(self.workingDir + outputFile, 'w')
    i = 0
    j = 0
    for k in records.keys():
      vi = Image.open(k)
      out.write('# ' + str(i) + '\n' + k + '\n3\n')
      out.write(str(vi.size[0]) + '\n' + str(vi.size[1]) + '\n')
      out.write(str(len(records[k])) + '\n')
      for d in records[k]:
        out.write(str(int(d[0])) + ' ' + "{:5.3f}".format(d[1])+ ' ' + "{:5.3f}".format(d[2]) + ' ')
        out.write(' '.join(map(str, map(int, d[3:]))) + '\n')
        j += 1
      i += 1
    out.close()
    return j

