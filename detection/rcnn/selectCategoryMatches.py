import os,sys
import libDetection as det
import utils as cu
import evaluation as eval

def selectBestBoxes(detections, groundTruth, minOverlap):
  candidates = []
  for d in detections:
    try: boxes = groundTruth[d[0]]
    except: continue
    bestIoU = 0.0
    for gt in boxes:
      iou = det.IoU(d[2:6],gt)
      if iou > bestIoU:
        bestIoU = iou
    print bestIoU
    if bestIoU > minOverlap:
      candidates.append(d)
  return candidates

def saveCandidates(candidates, output):
  out = open(output, 'w')
  for k in candidates:
    out.write(k[0] + ' ' + ' '.join(map(str, map(int, k[2:6]) ) ) + '\n')
  out.close()

if __name__ == "__main__":
  params = cu.loadParams("detectionsFile groundTruths output")
  detectionsData = [x.split() for x in open(params['detectionsFile'])]
  detections = eval.loadDetections(detectionsData)
  groundTruth = cu.loadBoxIndexFile(params['groundTruths'])
  candidates = selectBestBoxes(detections, groundTruth, 0.5)
  print 'Selected candidates:', len(candidates)
  saveCandidates(candidates, params['output'])
