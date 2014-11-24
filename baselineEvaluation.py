import os,sys
import utils as cu
import libDetection as det
import evaluation as eval

import json
import scipy.io
import numpy as np

def getCategories():
  categories = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  relations = [c + '_boxes' for c in categories] + [c + '_big' for c in categories] + [c + '_inside' for c in categories]
  relations.sort()
  categoryIndex = [ i for i in range(len(relations)) if relations[i].find('_boxes') != -1 ]
  return categories, categoryIndex

class CenterToEdgesDetector():
  def __init__(self, maxTime, categoryIndex):
    self.maxTime = maxTime
    self.categoryIndex = categoryIndex

  def run(self,img,features,bboxes):
    #boxSet = [map(float,b[1:]) for b in bboxes]
    boxSet = bboxes
    centers = [ ((b[2]-b[0])/2.0,(b[3]-b[1])/2.0) for b in boxSet]
    Cx,Cy = max([ (b[2],b[3]) for b in boxSet])
    Cx = Cx/2.0
    Cy = Cy/2.0
    euclid = lambda x: (x[0]-Cx)**2 + (x[1]-Cy)**2
    dist = map(euclid, centers)
    rank = np.argsort(dist)
    rankedBoxes = [boxSet[i].tolist() for i in rank[0:self.maxTime]]
    scores = [features[i,self.categoryIndex].tolist() for i in rank[0:self.maxTime]]
    return (img, rankedBoxes, scores, range(len(rankedBoxes)))

class BigestToSmallestArea():
  def __init__(self, maxTime, categoryIndex):
    self.maxTime = maxTime
    self.categoryIndex = categoryIndex

  def run(self, img, features, bboxes):
    boxSet = bboxes
    areas = [det.area(b) for b in boxSet]
    rank = np.argsort(areas)[::-1]
    rankedBoxes = [boxSet[i].tolist() for i in rank[0:self.maxTime]]
    scores = [features[i,self.categoryIndex].tolist() for i in rank[0:maxTime]]
    return (img, rankedBoxes, scores, range(len(rankedBoxes)))

class Objectness():
  def __init__(self, maxTime, categoryIndex):
    self.maxTime = maxTime
    self.categoryIndex = categoryIndex

  def run(self, img, features, bboxes):
    fp = lambda x: '_'.join(map(str,map(int,x)))
    boxIndex = dict([ (fp(bboxes[i]),i) for i in range(len(bboxes)) ])
    data = scipy.io.loadmat('/home/caicedo/data/relationsRCNN/objectness/'+img+'.mat')
    boxSet = [data['R'][i,0:4].tolist() for i in range(data['R'].shape[0]) ]
    objectness = [data['R'][i,4] for i in range(data['R'].shape[0])]
    rank = np.argsort(objectness)[::-1]
    rankedBoxes = [boxSet[i] for i in rank[0:self.maxTime]]
    scoresIndex = [ boxIndex[k] for k in map(fp, rankedBoxes) ]
    scores = [features[i,self.categoryIndex].tolist() for i in scoresIndex]
    return (img, rankedBoxes, scores, range(len(rankedBoxes)))
    
def loadDetections(imageList, featuresDir, ranking):
  totalNumberOfBoxes = 0
  sumOfPercentBoxesUsed = 0
  totalImages = len(imageList)
  scoredDetections = {}
  for imageName in imageList:
    features = scipy.io.loadmat(featuresDir + imageName + '.mat')
    img, boxes, scores, time = ranking.run(imageName, features['scores'], features['boxes'])
    scoredDetections[imageName] = {'boxes':boxes, 'scores':scores, 'time':time}
    totalNumberOfBoxes += len(boxes)
    percentBoxesUsed = 100*(float(len(boxes))/features['boxes'].shape[0])
    sumOfPercentBoxesUsed += percentBoxesUsed
    print imageName,'boxes:',len(boxes),'({:5.2f}% of {:4})'.format(percentBoxesUsed, features['boxes'].shape[0]),'scores:',len(scores)
    #if totalImages > 5: break
  print 'Average boxes per image: {:5.1f}'.format(totalNumberOfBoxes/float(totalImages))
  print 'Average percent of boxes used: {:5.2f}%'.format(sumOfPercentBoxesUsed/float(totalImages))
  return scoredDetections

def evaluateCategory(scoredDetections, categoryIdx, maxTime, step, groundTruthFile, output):
  performance = []
  ## Do a time analysis evaluation
  T = 0
  for t in range(0,maxTime,step):
    print " ****************** TIME: {:3} ********************** ".format(T)
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
    T += 1
  return performance

def saveTimeResults(categories, results, outputFile):
  out = open(outputFile,'w')
  out.write(' '.join(categories) + '\n')
  for i in range(results.shape[0]):
    r = results[i,:].tolist()
    out.write(' '.join(map(str,r)) + '\n')

if __name__ == "__main__":
  params = cu.loadParams('relationFeaturesDir imageList maxTime groundTruthDir outputDir category')
  images = [x.strip() for x in open(params['imageList'])]
  maxTime = int(params['maxTime'])
  step = 3
  categories, categoryIndex = getCategories()
  #ranking = CenterToEdgesDetector(maxTime, categoryIndex)
  #ranking = BigestToSmallestArea(maxTime, categoryIndex)
  ranking = Objectness(maxTime, categoryIndex)
  scoredDetections = loadDetections(images, params['relationFeaturesDir'], ranking)
  
  P = np.zeros( (maxTime/step, len(categories)) )
  R = np.zeros( (maxTime/step, len(categories)) )

  if params['category'] == 'all':
    catIdx = range(len(categories))
  else:
    catIdx = [i for i in range(len(categories)) if categories[i] == params['category']]

  for i in catIdx:
    groundTruthFile = params['groundTruthDir'] + '/' + categories[i] + '_test_bboxes.txt'
    outputFile = params['outputDir'] + '/' + categories[i] + '.out'
    performance = evaluateCategory(scoredDetections, i, maxTime, step, groundTruthFile, outputFile)
    P[:,i] = np.array( [p[0] for p in performance] )
    R[:,i] = np.array( [r[1] for r in performance] )
  saveTimeResults(categories, P, params['outputDir'] + '/precision_through_time.txt')
  saveTimeResults(categories, R, params['outputDir'] + '/recall_through_time.txt')
