import os
import utils as cu
import libLearning as learn
import libDetection as det
import trainDetector as training
import evaluation
import numpy as np

minOverlap = 0.8

########################################
## POSITIVES FROM TOP DETECTIONS
########################################
def buildDataSetWithTopDetections(detectionsData,topKPositives,featuresDir,featuresExt):
  detections = evaluation.loadDetections(detectionsData)
  m = cu.loadMatrix(featuresDir+'/'+detections[0][0]+'.'+featuresExt)
  positivesInfo = {}
  for i in range(topKPositives):
    d = detections[i]
    try:
      positivesInfo[ d[0] ].append( map(int,d[2:6]) )
    except:
      positivesInfo[ d[0] ] = [ map(int,d[2:6]) ]
  pos,posIdx = learn.loadHardNegativesFromList(featuresDir,positivesInfo,featuresExt,m.shape[1],topKPositives)
  #ari,osi = learn.computeAspectRatioAndSizeIntervals(posIdx)
  ari,osi = [],[]
  print 'Positive Matrix with Top Detections ('+str(pos.shape[0])+' instances)'
  return (pos,posIdx,ari,osi)

########################################
## POSITIVES WITH HIGH OVERLAP
########################################
def buildDataSetWithHighOverlap(positivesFeatures,logData,topKPositives,featuresDir,featuresExt):
  allPos,allPosIdx,ari,osi = training.readPositivesData(positivesFeatures)
  detections = [ x[0:5]+[float(x[6]),x[7]] for x in logData[0:2*topKPositives] ]
  detections = [ x for x in detections if x[6]=='1' and x[5] >= minOverlap ]
  pos = np.zeros( (len(detections),allPos.shape[1]) )
  posIdx = []
  for i in range(len(detections)):
    imgName = detections[i][0]
    box = map(float,detections[i][1:5])
    bestOverlap,bestIdx,idx = 0.0,0,0
    for truePos in allPosIdx:
      if truePos[0] == imgName:
        ov = det.IoU(box,map(float,truePos[1:5]))
        if ov > bestOverlap:
          bestOverlap = ov
          bestIdx = idx
      idx += 1
    pos[i,:] = allPos[bestIdx,:]
    posIdx.append(allPosIdx[bestIdx])
  print 'Positive Matrix with High Overlaping Detections ('+str(pos.shape[0])+' instances)'
  return (pos,posIdx,ari,osi)

########################################
## MAIN PROGRAM
########################################
if __name__ == "__main__":
  params = cu.loadParams('detectionsFile logFile topKParameter trainingList featuresDir featuresExt modelOut cost maxNegOverlap iterations')
  trainingList = [x.replace('\n','') for x in open(params['trainingList'])]
  cost = float(params['cost'])
  maxNegOverlap = float(params['maxNegOverlap'])
  topK = int(params['topKParameter'])
  if os.path.isfile(params['logFile']):
    # Positives with High Overlap can be found in log file
    logData = [x.split() for x in open(params['logFile'])]
    positives = buildDataSetWithHighOverlap(params['detectionsFile'],logData,topK,params['featuresDir'],params['featuresExt'])
  else:
    # Positives from top detections
    detectionsData = [x.split() for x in open(params['detectionsFile'])]
    positives = buildDataSetWithTopDetections(detectionsData,topK,params['featuresDir'],params['featuresExt'])
  # Run the training algorithm
  iterations = int(params['iterations'])+1
  for i in range(iterations):
    training.mainLoop(positives,trainingList,params['featuresDir'],params['featuresExt'],params['modelOut'],cost,maxNegOverlap,i)

