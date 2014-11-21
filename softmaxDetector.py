import os,sys
import utils as cu
import libDetection as det
import BoxSearchEvaluation as bse
from dataProcessor import processData

class SoftmaxDetector():

  def __init__(self, maxOverlap, categoryIndex):
    self.maxOverlap = maxOverlap
    self.categoryIndex = categoryIndex

  def run(self, image, features, boxes):
    print image, features.shape
    result = {}
    boxSet = [ map(float, b[1:]) for b in boxes]
    for i in self.categoryIndex:
      scores = features[:,i]
      fb,fs = det.nonMaximumSuppression(boxSet, scores, self.maxOverlap)
      result[i] = (image, fb, fs)
    return result

########################################
## RUN OBJECT DETECTOR
########################################
def detectObjects(imageList, featuresDir, indexType, groundTruthDir, outputDir):
  maxOverlap = 0.3
  categories, categoryIndex = bse.categoryIndex(indexType)
  task = SoftmaxDetector(maxOverlap, categoryIndex)
  result = processData(imageList, featuresDir, 'prob', task)
  # Collect detection results after NMS
  detections = dict([ (c,[]) for c in categoryIndex])
  for res in result:
    for idx in categoryIndex:
      for d in res[idx]:
        img, filteredBoxes, filteredScores = d
        for j in range(len(filteredBoxes)):
          detections[idx].append( [img, filteredScores[j]] + filteredBoxes[j] )
  # Evaluate results for each category independently
  for idx in categoryIndex:
    groundTruthFile = groundTruthDir + '/' + categories[idx] + '_test_bboxes.txt'
    output = outputDir + '/' + categories[i] + '.out'
    detections[idx].sort(key=lambda x:x[1], reverse=True)
    gtBoxes = [x.split() for x in open(groundTruthFile)]
    numPositives = len(gtBoxes)
    groundTruth = eval.loadGroundTruthAnnotations(gtBoxes)
    results = eval.evaluateDetections(groundTruth, detections[idx], 0.5)
    prec, recall = eval.computePrecisionRecall(numPositives, results['tp'], results['fp'], output)

if __name__ == "__main__":
  params = cu.loadParams('imageList scoresDir indexType groundTruthDir outputDir')
  imageList = [x.strip() for x in open(params['imageList'])]
  print 'Ready to process',len(imageList)
  detectObjects(imageList, params['scoresDir'], params['indexType'], params['groundTruthDir'], params['outputDir'])
