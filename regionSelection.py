import os,sys
import utils as cu
import libDetection as det
import numpy as np
from dataProcessor import processData

# Windows that contain a ground truth box
def big(box,gt):
  ov = det.overlap(box,gt)
  iou = det.IoU(box,gt)
  return ov >= 1.0 and iou < 0.5

# Windows that pass the PASCAL criteria
def tight(box,gt):
  iou = det.IoU(box,gt)
  return iou >= 0.8

# Windows inside a bounding box
def inside(box,gt):
  ov = det.overlap(gt,box)
  iou = det.IoU(box,gt)
  return ov >= 1.0 and iou <= 0.4 and iou >= 0.2

# Background windows
def background(box,gt):
  ov = det.overlap(box,gt)
  iou = det.IoU(box,gt)
  return ov < 0.3 and iou < 0.3

class RegionSelector():
  def __init__(self,groundTruths,operator):
    self.groundTruths = groundTruths
    self.operator = operator

  def run(self,img,features,bboxes):
    if not img in self.groundTruths.keys():
      return 0
    candidates = []
    index = []
    for b in bboxes:
      box = map(float,b[1:])
      match = False
      for gt in self.groundTruths[img]:
        match = self.operator(box,gt)
        if match:
          break
      if match:
        candidates.append(True)
        index.append( [img] + box )
      else:
        candidates.append(False)
    candidates = np.asarray(candidates)
    return (features[candidates],index)

def selectRegions(imageList, featuresDir, groundTruths, outputDir, featExt, category, operator):
  task = RegionSelector(groundTruths, operator)
  result = processData(imageList, featuresDir, featExt, task)
  nBoxes,nFeat = 0,0
  for r in result:
    nBoxes += r[0].shape[0]
    nFeat = r[0].shape[1]
  featureMatrix = np.zeros( (nBoxes,nFeat) )
  i = 0
  outputFile = open(outputDir + '/' + category + '.idx','w')
  for r in result:
    featureMatrix[i:i+r[0].shape[0]] = r[0]
    for box in r[1]:
      outputFile.write(box[0] + ' ' + ' '.join(map(str,map(int,box[1:]))) + '\n')
    i += r[0].shape[0]
  outputFile.close()
  cu.saveMatrixNoCompression(featureMatrix,outputDir + '/' + category + '.' + featExt)
  print 'Total of',nBoxes,'positive examples collected for',category

if __name__ == "__main__":
  params = cu.loadParams("imageList featuresDir groundTruthFile outputDir featuresExt category operation")
  groundTruths = cu.loadBoxIndexFile(params['groundTruthFile'])
  imageList = [x.replace('\n','') for x in open(params['imageList'])]
  operator = None
  if params['operation'] == 'big':
    operator = big
  elif params['operation'] == 'tight':
    operator = tight
  elif params['operation'] == 'inside':
    operator = inside
  elif params['operation'] == 'background':
    operator = background
  else:
    print 'Select a valid operation: [big | tight | inside | background]'
    sys.exit()

  selectRegions(imageList, params['featuresDir'], groundTruths, params['outputDir'], params['featuresExt'], params['category'], operator)
