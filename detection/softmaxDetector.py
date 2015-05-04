import os,sys
import utils as cu
import libDetection as det
import BoxSearchEvaluation as bse
from dataProcessor import processData
import evaluation as eval

class SoftmaxDetector():

  def __init__(self, maxOverlap, catIndex):
    self.maxOverlap = maxOverlap
    self.catIndex = catIndex

  def run(self, image, features, boxes):
    s = cu.tic()
    result = {}
    boxSet = [ map(float, b[1:]) for b in boxes ]
    for i in self.catIndex:
      scores = features[:,i]
      fb,fs = det.nonMaximumSuppression(boxSet, scores, self.maxOverlap)
      result[i] = (image, fb, fs)
    s = cu.toc(image, s)
    return result

########################################
## RUN OBJECT DETECTOR
########################################
def detectObjects(imageList, featuresDir, indexType, groundTruthDir, outputDir):
  maxOverlap = 0.3
  categories, catIndex = bse.categoryIndex(indexType)
  task = SoftmaxDetector(maxOverlap, catIndex)
  result = processData(imageList, featuresDir, 'prob', task)
  # Collect detection results after NMS
  detections = dict([ (c,[]) for c in catIndex])
  for res in result:
    for idx in catIndex:
      img, filteredBoxes, filteredScores = res[idx]
      for j in range(len(filteredBoxes)):
        detections[idx].append( [img, filteredScores[j]] + filteredBoxes[j] )
  # Evaluate results for each category independently
  for idx in catIndex:
    groundTruthFile = groundTruthDir + '/' + categories[idx] + '_test_bboxes.txt'
    output = outputDir + '/' + categories[idx] + '.out'
    detections[idx].sort(key=lambda x:x[1], reverse=True)
    gtBoxes = [x.split() for x in open(groundTruthFile)]
    numPositives = len(gtBoxes)
    groundTruth = eval.loadGroundTruthAnnotations(gtBoxes)
    results = eval.evaluateDetections(groundTruth, detections[idx], 0.5)
    prec, recall = eval.computePrecisionRecall(numPositives, results['tp'], results['fp'], output)

if __name__ == "__main__":
  params = cu.loadParams('imageList scoresDir indexType groundTruthDir outputDir')
  imageList = [x.strip() for x in open(params['imageList'])]
  imageList = imageList
  print 'Ready to process',len(imageList)
  detectObjects(imageList, params['scoresDir'], params['indexType'], params['groundTruthDir'], params['outputDir'])
