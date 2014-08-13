import os,sys
from sklearn import svm
import numpy as np
import utils as cu
import libLearning as learn
import libDetection as det

########################################
## IMPLEMENTATION OF LINEAR DETECTOR
########################################
class LinearDetector(det.Detector):
  def __init__(self,params=None):
    if params == None:
      params = {'C':1.0}
    self.C = float(params['C'])
    self.clf = svm.LinearSVC(C=self.C, class_weight='auto', loss='l2',random_state=cu.randomSeed,tol=cu.tolerance)

  def learn(self,pos,neg,posIdx=None,negIdx=None):
    Y = np.concatenate( (cu.posOnes([pos.shape[0]]), cu.negOnes([neg.shape[0]])) )
    self.clf.fit(np.concatenate( (pos,neg) ), Y)

  def predict(self,X,Z=None):
    return self.clf.decision_function(X)

  def predictAll(self,X,Z=None):
    return self.predict(X,Z),np.zeros((X.shape[0]))

  def save(self,outFile):
    cu.saveModel(self,outFile)

  def load(self,inputFile):
    svm_ = cu.loadModel(inputFile)
    self.C = svm_.C
    self.clf = svm_.clf

  def evaluate(self,X,Y):
    import sklearn.metrics as met
    pred = self.clf.predict(X)
    cfm = met.confusion_matrix(Y,pred)
    precision = float(cfm[1][1])/(cfm[1][1] + cfm[0][1])
    recall = float(cfm[1][1])/(cfm[1][1] + cfm[1][0])
    print '{:.4f} {:.4f}'.format(precision,recall)
    print cfm

