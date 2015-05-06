import os,sys
from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np
import utils as cu
import libLearning as learn
import libDetection as det
import trainDetector

########################################
## IMPLEMENTATION OF SINGLE LINEAR DETECTOR
########################################
class SingleDetector(det.Detector):
  def __init__(self,params=None):
    if params == None:
      params = {'C':1.0,'subcategoryLabels':'','subcategory':0}
    self.C = float(params['C'])
    self.clf = svm.LinearSVC(C=self.C, class_weight='auto', loss='l2',random_state=cu.randomSeed,tol=cu.tolerance)
    self.subcategory = int(params['subcategory'])
    self.subcategoryLabels = np.asarray([ int(x.split()[0]) for x in open(params['subcategoryLabels']) ])

  def learn(self,pos,neg,posIdx,negIdx):
    posLabels = cu.negOnes([pos.shape[0]])
    posLabels[self.subcategoryLabels == self.subcategory] = 1.
    print 'Using',len(posLabels[self.subcategoryLabels == self.subcategory]),'positives for subcategory',self.subcategory
    Y = np.concatenate( (posLabels, cu.negOnes([neg.shape[0]])) )
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

########################################
## IMPLEMENTATION OF SUBCATEGORIES DETECTOR
########################################
class SubcategoriesDetector(det.Detector):
  def __init__(self,params=None):
    if params == None:
      params = {'C':1.0}
    self.cost = int(params['C'])

  def learn(self,pos,neg,posIdx,negIdx):
    pass

  def predictAll(self,X,Z):
    scores = -np.inf*np.ones( [X.shape[0],self.subcategories] )
    for k in range(self.subcategories):
      if self.classifiers[k] != None:
        scores[:,k] = self.classifiers[k].predict(X)
    return (np.max(scores,axis=1), np.argmax(scores,axis=1))

  def predict(self,X,Z=None):
    scores,latentLabels = self.predictAll(X,Z)
    return scores

  def save(self,outFile):
    pass

  def load(self,inputFile):
    print 'OPENING',inputFile
    if inputFile.find('*') != -1:
      self.classifiers = []
      i = 0
      file = inputFile.replace('*',str(i))
      while os.path.isfile(file):
        print 'Loading model',i
        self.classifiers.append(cu.loadModel(file))
        i += 1
        file = inputFile.replace('*',str(i))
      self.subcategories = i
    else:
      print 'WRONG FILE PATTER FOR SUBCATEGORIES MODEL:',inputFile


def initializeSubcategories(positives,numCategories,outputFile):
  pos,posIdx,ari,osi = positives
  clustering = KMeans(init='k-means++', n_clusters=numCategories, n_init=50,max_iter=1000,random_state=cu.randomSeed,tol=cu.tolerance)
  clustering.fit(pos)
  outf = open(outputFile,'w')
  for k in clustering.labels_:
    outf.write(str(k)+'\n')
  outf.close()
  print 'Clustering done'
  return clustering.labels_

if __name__ == '__main__':
  params = cu.loadParams("modelParams positivesFeatures trainingList featuresDir modelOut overlap iterations labelsFile totalNumberOfSubcategories subcategory")
  featuresExt = params['positivesFeatures'].split('.')[-1]
  trainingList = [x.replace('\n','') for x in open(params['trainingList'])]
  maxNegOverlap = float(params['overlap'])
  iterations = int(params['iterations'])+1
  positives = trainDetector.readPositivesData(params['positivesFeatures'])
  if not os.path.isfile(params['labelsFile']):
    initializeSubcategories(positives,int(params['totalNumberOfSubcategories']),params['labelsFile'])
  sys.exit()
  params['modelParams'] = params['modelParams']+'subcategoryLabels:'+params['labelsFile']+'!subcategory:'+params['subcategory']+'!'
  args = trainDetector.parseModelParams(params['modelParams'])
  print " ++ LEARNING SUBCATEGORIES MODEL WITH ARGS:",params['modelParams']," ++ "
  for i in range(iterations):
    trainDetector.mainLoop('single',args,positives,trainingList,params['featuresDir'],featuresExt,params['modelOut'],maxNegOverlap,i)
  os.system('rm '+params['modelOut']+'.hards')
  

