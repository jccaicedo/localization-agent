import os,sys
import sklearn
import numpy as np
import utils as cu
import libLearning as learn
import libDetection as det
import detector
import evaluation

## Program Variables
maxNegativeImages  = 100
maxVectorsCache    = 50000
maxNegativeVectors = 150000
np.random.seed( cu.randomSeed )

def mainLoop(modelType,modelArgs,positives,trainingList,featuresDir,featuresExt,modelOut,maxNegOverlap,iter):
  pos,posIdx,ari,osi = positives
  startTime = cu.tic()
  if iter == 0:
    ## Random Negatives
    print ' >>> RANDOM NEGATIVES'
    neg,negIdx = learn.getRandomNegs(featuresDir,trainingList,featuresExt,pos.shape[1],maxVectorsCache,maxNegativeImages)
    detectionsList = [ [x[0],'0.0']+x[1:]+['1'] for x in negIdx]
    hards = {'features':np.zeros((0,neg.shape[1])),'index':[]}
    lap = cu.toc('Random negatives matrix ('+str(neg.shape[0])+' instances)',startTime)
  else:
    ## Mine hard negatives
    print ' >>> MINING HARD NEGATIVES'
    model = det.createDetector(modelType,modelArgs)
    model.load(modelOut+'.'+ str( iter-1 ))
    detectionsList = detector.detectObjects(model,trainingList,featuresDir,featuresExt,1.0,-10.0) # For RCNN the overlap parameter is 0.3 not 1.0(no suppression)
    hards = cu.loadMatrixNoCompression(modelOut+'.hards').item()
    lap = cu.toc('Hard negatives matrix ('+str(hards['features'].shape[0])+' instances)',startTime)

  ## Rank and clean negative detections
  detectionsData = evaluation.loadDetections(detectionsList)
  groundTruth = evaluation.loadGroundTruthAnnotations(posIdx)
  detectionsLog = evaluation.evaluateDetections(groundTruth,detectionsData,0.5,allowDuplicates=False) #,overlapMeasure=det.overlap
  evaluation.computePrecisionRecall(len(posIdx),detectionsLog['tp'],detectionsLog['fp'],'tmp.txt')
  evaluation.computePrecAt(detectionsLog['tp'],[20,50,100,200,300,400,500])
  logData = learn.parseRankedDetectionsFile(detectionsLog['log'],maxNegOverlap,maxNegativeVectors)
  print ' >>> LOADING HARD NEGATIVES'
  neg,negIdx = learn.loadHardNegativesFromList(featuresDir,logData['negExamples'],featuresExt,pos.shape[1],logData['negTaken'])
  del(detectionsList,detectionsData,detectionsLog,logData)
  lap = cu.toc('Ranked negatives matrix ('+str(neg.shape[0])+' instances)',lap)
  neg = np.concatenate( (neg,hards['features']) )
  negIdx = negIdx + hards['index']

  ## Learn Detector
  clf = det.createDetector(modelType,modelArgs)
  clf.learn(pos,neg,posIdx,negIdx)
  clf.save(modelOut+'.'+str(iter))
  lap = cu.toc('Classifier learned:',lap)

  ## Keep hard negatives for next iterations
  scores = clf.predict(neg,negIdx)
  hardNegsIdx = np.argsort(scores)
  hardNeg = np.concatenate( (hards['features'], neg[hardNegsIdx[-cu.topHards:]]) )
  negIdx = hards['index'] + [negIdx[j] for j in hardNegsIdx[-cu.topHards:]]
  print 'Hard negatives:',hardNeg.shape[0]
  hards = {'features':hardNeg, 'index':negIdx}
  cu.saveMatrixNoCompression({'features':hardNeg,'index':negIdx},modelOut+'.hards')

  print ' ** Iteration',iter,'done'
  return {'detector':clf,'pos':pos,'posIdx':posIdx,'neg':neg,'negIdx':negIdx}

########################################
## READ POSITIVE DATA
########################################
def readPositivesData(positivesFeatures):
  pos,posIdx = cu.loadMatrixAndIndex(positivesFeatures)
  ari,osi = [],[]
  print 'Positive Matrix loaded ('+str(pos.shape[0])+' instances)'
  return (pos,posIdx,ari,osi)

def parseModelParams(params):
  # Expected Format: k:v!k:v!
  params = params.split('!')
  result = {}
  for p in params:
    if p != '':
      k,v = p.split(':')
      result[k] = v
  return result

########################################
## MAIN PROGRAM
########################################
if __name__ == "__main__":
  params = cu.loadParams("modelType modelParams positivesFeatures trainingList featuresDir modelOut overlap iterations")
  featuresExt = params['positivesFeatures'].split('.')[-1]
  trainingList = [x.replace('\n','') for x in open(params['trainingList'])]
  maxNegOverlap = float(params['overlap'])
  iterations = int(params['iterations'])+1
  positives = readPositivesData(params['positivesFeatures'])
  args = parseModelParams(params['modelParams'])
  print " ++ LEARNING",params['modelType'],"MODEL WITH ARGS:",params['modelParams']," ++ "
  for i in range(iterations):
    mainLoop(params['modelType'],args,positives,trainingList,params['featuresDir'],featuresExt,params['modelOut'],maxNegOverlap,i)
  os.system('rm '+params['modelOut']+'.hards')

