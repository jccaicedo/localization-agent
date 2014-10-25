import os,sys
import utils as cu
import libDetection as det
import numpy as np
from dataProcessor import processData
import Image

# Windows that contain a ground truth box
def big(box, gt, a=0.9, b=0.3, c=0.2):
  ov = det.overlap(box,gt)
  iou = det.IoU(box,gt)
  return ov >= a and iou <= b and iou >= c

# Windows that pass the PASCAL criteria
def tight(box, gt, a=1.0):
  iou = det.IoU(box,gt)
  return iou >= a

# Windows inside a bounding box
def inside(box, gt, a=1.0, b=0.3, c=0.2):
  ov = det.overlap(gt,box)
  iou = det.IoU(box,gt)
  return ov >= a and iou <= b and iou >= c

# Background windows
def background(box,gt):
  ov = det.overlap(box,gt)
  iou = det.IoU(box,gt)
  return ov < 0.3 and iou < 0.3

class RegionSelector():
  def __init__(self,groundTruths,operator,category):
    self.groundTruths = groundTruths
    self.operator = operator
    self.category = category + '_' + operator.__name__

  def run(self,img,bboxes):
    if not img in self.groundTruths.keys():
      return []
    candidates = []
    index = []
    for b in bboxes:
      box = map(float,b[:])
      match = False
      for gt in self.groundTruths[img]:
        match = self.operator(box,gt)
        if match:
          break
      if match:
        candidates.append(True)
        index.append( [img, self.category] + box )
      else:
        candidates.append(False)
    candidates = np.asarray(candidates)
    return index

def selectRegions(proposals, groundTruths, category, operator):
  task = RegionSelector(groundTruths, operator, category)
  result = []
  for img in proposals.keys():
    result += task.run(img, proposals[img])
  return result

def reformatGroundTruth(gt, category):
  result = []
  for img in gt.keys():
    for box in gt[img]:
      result.append( [img, category + '_tight'] + box )
  return result

def saveResults(outputFile, results):
  outputFile = open(outputFile,'w')
  for r in results:
    outputFile.write(r[0] + ' ' + r[1] + ' ' + ' '.join(map(str,map(int,r[2:]))) + '\n')
  outputFile.close()

if __name__ == "__main__":
  params = cu.loadParams("proposalsFile groundTruthDir outputFile")
  proposals = cu.loadBoxIndexFile(params['proposalsFile'])
  records = []
  files = os.listdir(params['groundTruthDir'])
  files.sort()
  for f in files:
    category = f.split('_')[0]
    print category
    groundTruth = cu.loadBoxIndexFile(params['groundTruthDir'] + '/' + f)
    records += selectRegions(proposals, groundTruth, category, big)
    records += selectRegions(proposals, groundTruth, category, inside)
    #records += reformatGroundTruth(groundTruth, category)
  saveResults(params['outputFile'], records)

