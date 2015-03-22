import utils as cu
import numpy as np

import RLConfig as config

class PriorMemory():

  def __init__(self, allObjectsFile, categoryObjects, convnet):
    self.allObjects  = cu.loadBoxIndexFile(allObjectsFile)
    self.categoryObjects = categoryObjects
    negativeSamples = reduce(lambda x,y:x+y, map(len,self.allObjects.values()))
    positiveSamples = reduce(lambda x,y:x+y, map(len,self.categoryObjects.values()))
    self.N = np.zeros( (negativeSamples, 4096), np.float32 )
    self.P = np.zeros( (positiveSamples, 4096), np.float32 )
    idx = 0
    # Populate negative examples
    print '# Processing',negativeSamples,'negative prior samples'
    for key in self.allObjects.keys():
      try:
        boxes = self.categoryObjects[key]
        cover = True
      except:
        boxes = self.allObjects[key]
        cover = False
      convnet.prepareImage(key)
      for box in boxes:
        if cover:
          convnet.coverRegion(box)
        activations = convnet.getActivations(box)
        self.N[idx,:] = activations[config.get('convnetLayer')]
        idx += 1
    # Populate positive examples
    print '# Processing',positiveSamples,'positive prior samples'
    idx = 0
    for key in self.categoryObjects.keys():
      convnet.prepareImage(key)
      for box in self.categoryObjects[key]:
        activations = convnet.getActivations(box)
        self.P[idx,:] = activations[config.get('convnetLayer')]

