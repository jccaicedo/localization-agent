import os,sys
import utils as cu
import numpy as np
import libDetection as det
from dataProcessor import processData

class IntegratedModel():
  def __init__(self, dir):
    allModels = []
    for f in os.listdir(dir):
      model = det.createDetector('linear')
      model.load(dir + f)
      allModels.append(model)
    feat = allModels[0].clf.coef_.shape[1]
    M = np.zeros( (len(allModels), feat + 1) )
    for i in range(len(allModels)):
      M[i,0:feat] = allModels[i].clf.coef_[0,:]
      M[i,feat] = allModels[i].clf.intercept_[0]
    self.M = M

  def decisionFunction(self, X):
    F = np.ones( (X.shape[0], self.M.shape[1]) )
    F[:,0:-1] = X
    return np.dot(F, self.M.T)

  def sigmoidValues(self, X):
    return 1/(1 + np.exp(- self.decisionFunction(X) ))

class CategoryScores():
  def __init__(self, integratedModel, outDir):
    self.model = integratedModel
    self.outDir = outDir

  def run(self,img,features,bboxes):
    print img
    scores = self.model.sigmoidValues(features)
    cu.saveMatrix(scores, self.outDir+'/'+img+'.sigmoid_scores')
    return

def extractFeatures(model,imageList,featuresDir,featuresExt):
  task = CategoryScores(model,featuresDir)
  result = processData(imageList,featuresDir,featuresExt,task)

if __name__ == "__main__":
  params = cu.loadParams('modelsDir imageList featuresDir featuresExt')
  im = IntegratedModel(params['modelsDir'])
  imageList = [x.replace('\n','') for x in open(params['imageList'])]
  extractFeatures(im, imageList, params['featuresDir'], params['featuresExt'])

