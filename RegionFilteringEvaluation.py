import os,sys
import utils as cu
import libDetection as det
import evaluation as eval

import json
import scipy.io
import numpy as np

params = cu.loadParams('testMemDir relationFeaturesDir groundTruthFile category output')

categories = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
categories = [c + '_boxes' for c in categories] + [c + '_big' for c in categories] + [c + '_inside' for c in categories]
categories.sort()
categories = dict( [ (categories[i],i) for i in range(len(categories)) ] )
categoryIndex = categories[params['category']+'_boxes']

totalNumberOfBoxes = 0
sumOfPercentBoxesUsed = 0
totalImages = 0
scoredDetections = {}
for f in os.listdir(params['testMemDir']):
  imageName = f.replace('.txt','')
  totalImages += 1
  data = json.load( open(params['testMemDir']+f, 'r') )
  features = scipy.io.loadmat(params['relationFeaturesDir'] + imageName + '.mat')
  boxes = []
  scores = []
  time = []
  t = 0
  for boxSet in data['boxes']:
    for box in boxSet:
      score = np.NaN
      # Find scores for this box
      for i in range(features['boxes'].shape[0]):
        iou = det.IoU(box, features['boxes'][i,:].tolist())
        if iou == 1.0:
          scores.append( features['scores'][i,categoryIndex] )
          break
      boxes.append( box )
      time.append( t )
    t += 1
  scoredDetections[imageName] = {'boxes':boxes, 'scores':scores, 'time':time}
  totalNumberOfBoxes += len(boxes)
  percentBoxesUsed = 100*(float(len(boxes))/features['boxes'].shape[0])
  sumOfPercentBoxesUsed += percentBoxesUsed
  print imageName,'boxes:',len(boxes),'({:5.2f}% of {:4})'.format(percentBoxesUsed, features['boxes'].shape[0]),'scores:',len(scores)

maxTime = np.max(time)
print 'Average boxes per image: {:5.1f}'.format(totalNumberOfBoxes/float(totalImages))
print 'Average percent of boxes used: {:5.2f}%'.format(sumOfPercentBoxesUsed/float(totalImages))

## Do a time analysis evaluation
for t in range(maxTime):
  print " ****************** TIME: {:3} ********************** ".format(t) 
  detections = []
  for img in scoredDetections.keys():
    data = scoredDetections[img]
    idx = [i for i in range(len(data['time'])) if data['time'][i] <= t]
    boxes = [data['boxes'][i] for i in idx]
    scores = [data['scores'][i] for i in idx]
    if len(boxes) > 0:
      fBoxes, fScores = det.nonMaximumSuppression(boxes, scores, 0.3)
      for i in range(len(fBoxes)):
        detections.append( [img, fScores[i]] + fBoxes[i] )
  detections.sort(key=lambda x:x[1], reverse=True)
  gtBoxes = [x.split() for x in open(params['groundTruthFile'])]
  numPositives = len(gtBoxes)
  groundTruth = eval.loadGroundTruthAnnotations(gtBoxes)
  results = eval.evaluateDetections(groundTruth, detections, 0.5)
  eval.computePrecisionRecall(numPositives, results['tp'], results['fp'], params['output'])

