import os,sys
import utils.utils as cu
import utils.libDetection as det
import detection.evaluation as eval

import json
import scipy.io
import numpy as np

def categoryIndex(type):
  categories, catIndex = [],[]
  if type == 'pascal':
    categories, catIndex = get20Categories()
  elif type == 'relations':
    categories, catIndex = getCategories()
  elif type == 'finetunedRelations':
    categories, catIndex = getRelationCategories()
  return categories, catIndex

def getCategories():
  categories = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  relations = [c + '_boxes' for c in categories] + [c + '_big' for c in categories] + [c + '_inside' for c in categories]
  relations.sort()
  catIndex = [ i for i in range(len(relations)) if relations[i].find('_boxes') != -1 ]
  return categories, catIndex

def get20Categories():
  categories = 'background aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  catIndex = [ i for i in range(1,len(categories)) ]
  return categories, catIndex

def getRelationCategories():
  categories = 'background aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  relations = categories + [categories[i+1]+'_big' for i in range(len(categories)-1)] + [categories[i+1]+'_inside' for i in range(len(categories)-1)]
  catIndex = [ i for i in range(1,len(categories)) ]
  return categories, catIndex

def loadScores(memDir, catI):
  totalNumberOfBoxes = 0
  sumOfPercentBoxesUsed = 0
  totalImages = 0
  scoredDetections = {}
  for f in os.listdir(memDir):
    if not f.endswith('.txt'): continue
    imageName = f.replace('.txt','')
    totalImages += 1
    data = json.load( open(memDir + f, 'r') )
    boxes = []
    scores = []
    values = []
    landmarks = []
    t = 0
    for i in range(len(data['boxes'])):
      boxes.append( data['boxes'][i] )
      if catI > 0:
        scores.append( data['scores'][i][catI])
      else:
        scores.append( 0 )
      values.append( data['values'][i] )
      if data['actions'][i] == 8:
        landmarks.append( data['values'][i] )
      else:
        landmarks.append( float('-inf') )
      totalNumberOfBoxes += 1
      t += 1
    scoredDetections[imageName] = {'boxes':boxes, 'scores':scores, 'values':values, 'landmarks':landmarks}
    print imageName,'detections:',len(boxes)
    #if totalImages > 5: break

  print 'Average boxes per image: {:5.1f}'.format(totalNumberOfBoxes/float(totalImages))
  return scoredDetections

def evaluateCategory(scoredDetections, ranking, groundTruthFile, output=None):
  performance = []
  detections = []
  for img in scoredDetections.keys():
    data = scoredDetections[img]
    idx = range(len(data['boxes']))
    boxes = [data['boxes'][i] for i in idx if data[ranking][i] > float('-inf')]
    scores = [data[ranking][i] for i in idx if data[ranking][i] > float('-inf')]
    if len(boxes) > 0:
      fBoxes, fScores = det.nonMaximumSuppression(boxes, scores, 0.3)
      for i in range(len(fBoxes)):
        detections.append( [img, fScores[i]] + fBoxes[i] )
  detections.sort(key=lambda x:x[1], reverse=True)
  gtBoxes = [x.split() for x in open(groundTruthFile)]
  numPositives = len(gtBoxes)
  groundTruth = eval.loadGroundTruthAnnotations(gtBoxes)
  results = eval.evaluateDetections(groundTruth, detections, 0.5)
  if output is not None:
    output = output + '.' + ranking
  prec, recall = eval.computePrecisionRecall(numPositives, results['tp'], results['fp'], output)
  return prec, recall

if __name__ == "__main__":
  params = cu.loadParams('testMemDir groundTruthFile outputDir')

  scoredDetections = loadScores(params['testMemDir'], -1)
  
  groundTruthFile = params['groundTruthFile']
  outputFile = params['outputDir'] + '/' + 'result.out'
  pl,rl = evaluateCategory(scoredDetections, 'landmarks', groundTruthFile, outputFile)
  line = lambda x,y,z: x + '\t{:5.3f}\t{:5.3f}\n'.format(y,z)
  out = open(params['outputDir'] + '/evaluation.txt','w')
  out.write('\tPrecision\tRecall\n')
  out.write(line('Landmarks',pl,rl))
  out.close()
