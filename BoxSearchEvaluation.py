import os,sys
import utils as cu
import libDetection as det
import evaluation as eval

import json
import scipy.io
import numpy as np

def categoryIndex(type):
  categories, categoryIndex = [],[]
  if type == 'pascal':
    categories, categoryIndex = get20Categories()
  elif type == 'relations':
    categories, categoryIndex = getCategories()
  elif type == 'finetunedRelations':
    categories, categoryIndex = getRelationCategories()
  return categories, categoryIndex

def getCategories():
  categories = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  relations = [c + '_boxes' for c in categories] + [c + '_big' for c in categories] + [c + '_inside' for c in categories]
  relations.sort()
  categoryIndex = [ i for i in range(len(relations)) if relations[i].find('_boxes') != -1 ]
  return categories, categoryIndex

def get20Categories():
  categories = 'background aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  categoryIndex = [ i for i in range(1,len(categories)) ]
  return categories, categoryIndex

def getRelationCategories():
  categories = 'background aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  relations = categories + [categories[i+1]+'_big' for i in range(len(categories)-1)] + [categories[i+1]+'_inside' for i in range(len(categories)-1)]
  cateogryIndex = [ i for i in range(1,len(categories)) ]
  return categories, categoryIndex

def loadScores(memDir, categoryIndex):
  print categoryIndex
  totalNumberOfBoxes = 0
  sumOfPercentBoxesUsed = 0
  totalImages = 0
  scoredDetections = {}
  for f in os.listdir(memDir):
    imageName = f.replace('.txt','')
    totalImages += 1
    data = json.load( open(memDir + f, 'r') )
    boxes = []
    scores = []
    time = []
    t = 0
    for i in range(len(data['boxes'])):
      scores.append( [data['scores'][i][j] for j in categoryIndex ] )
      boxes.append( data['boxes'][i] )
      time.append( t )
      t += 1
    scoredDetections[imageName] = {'boxes':boxes, 'scores':scores, 'time':time}
    totalNumberOfBoxes += len(boxes)
    print imageName,'boxes:',len(boxes)
    #if totalImages > 5: break

  maxTime = np.max(time)
  print 'Average boxes per image: {:5.1f}'.format(totalNumberOfBoxes/float(totalImages))
  print 'Average percent of boxes used: {:5.2f}%'.format(sumOfPercentBoxesUsed/float(totalImages))
  return scoredDetections, maxTime

def evaluateCategory(scoredDetections, categoryIdx, maxTime, groundTruthFile, output):
  performance = []
  ## Do a time analysis evaluation
  for t in range(maxTime):
    print " ****************** TIME: {:3} ********************** ".format(t) 
    detections = []
    for img in scoredDetections.keys():
      data = scoredDetections[img]
      idx = [i for i in range(len(data['time'])) if data['time'][i] <= t]
      boxes = [data['boxes'][i] for i in idx]
      scores = [data['scores'][i][categoryIdx] for i in idx]
      if len(boxes) > 0:
        fBoxes, fScores = det.nonMaximumSuppression(boxes, scores, 0.3)
        for i in range(len(fBoxes)):
          detections.append( [img, fScores[i]] + fBoxes[i] )
    detections.sort(key=lambda x:x[1], reverse=True)
    gtBoxes = [x.split() for x in open(groundTruthFile)]
    numPositives = len(gtBoxes)
    groundTruth = eval.loadGroundTruthAnnotations(gtBoxes)
    results = eval.evaluateDetections(groundTruth, detections, 0.5)
    if t == maxTime - 1:
      prec, recall = eval.computePrecisionRecall(numPositives, results['tp'], results['fp'], output)
    else:
      prec, recall = eval.computePrecisionRecall(numPositives, results['tp'], results['fp'])
    performance.append( [prec, recall] )
  return performance

def saveTimeResults(categories, results, outputFile):
  out = open(outputFile,'w')
  out.write(' '.join(categories) + '\n')
  for i in range(results.shape[0]):
    r = results[i,:].tolist()
    out.write(' '.join(map(str,r)) + '\n')

if __name__ == "__main__":
  params = cu.loadParams('testMemDir groundTruthDir outputDir category categoryIndex')
  if params['categoryIndex'] == 'pascal':
    categories, categoryIndex = get20Categories()
  elif params['categoryIndex'] == 'relations':
    categories, categoryIndex = getCategories()
  elif params['categoryIndex'] == 'finetunedRelations':
    categories, categoryIndex = getRelationCategories()
  scoredDetections, maxTime = loadScores(params['testMemDir'], categoryIndex)
  
  P = np.zeros( (maxTime, len(categories)) )
  R = np.zeros( (maxTime, len(categories)) )

  if params['category'] == 'all':
    catIdx = range(len(categories))
  else:
    catIdx = [i for i in range(len(categories)) if categories[i] == params['category']]

  for i in catIdx:
    groundTruthFile = params['groundTruthDir'] + '/' + categories[i] + '_test_bboxes.txt'
    outputFile = params['outputDir'] + '/' + categories[i] + '.out'
    performance = evaluateCategory(scoredDetections, i, maxTime, groundTruthFile, outputFile)
    P[:,i] = np.array( [p[0] for p in performance] )
    R[:,i] = np.array( [r[1] for r in performance] )
  saveTimeResults(categories, P, params['outputDir'] + '/precision_through_time.txt')
  saveTimeResults(categories, R, params['outputDir'] + '/recall_through_time.txt')
