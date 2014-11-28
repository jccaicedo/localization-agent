__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task
import BoxSearchState as bss

import utils as cu
import libDetection as det
import numpy as np

minAcceptableIoU = 0.5

def center(box):
  return [ (box[2] + box[0])/2.0 , (box[3] + box[1])/2.0 ]

def euclideanDist(c1, c2):
  return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2

class BoxSearchTask(Task):

  def __init__(self, environment=None, groundTruthFile=None):
    Task.__init__(self, environment)
    if groundTruthFile is not None:
      self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)
    self.image = ''
    self.epochRecall = []
    self.epochMaxIoU = []

  def getReward(self):
    gt = self.loadGroundTruth(self.env.imageList[self.env.idx])
    reward = self.computeObjectReward(self.env.state.box, self.env.state.actionChosen, self.env.state.visitedBefore())
    self.env.updatePostReward(reward)
    return reward

  def computeObjectRewardV4(self, box, actionChosen, visitedBefore, update=True):
    iou, idx = self.matchBoxes(box)
    improvedIoU = 0.0
    if iou > self.control['IOU'][idx]:
      if update: self.control['IOU'][idx] = iou
      improvedIoU += 1.0
    wellLocalizedObject = 0.0
    if iou >= minAcceptableIoU:
      wellLocalizedObject += 1.0
    if actionChosen == bss.PLACE_LANDMARK and iou >= 0.8:
      wellLocalizedObject += 2.0
    visibleObject = 0.0
    if iou == 0.0:
      visibleObject = -2.0
    return improvedIoU + wellLocalizedObject + visibleObject

  def computeObjectRewardV3(self, box, actionChosen, visitedBefore, update=True):
    iou, idx = self.matchBoxes(box)
    improvedIoU = 0.0
    if iou > self.control['IOU'][idx]:
      if update: self.control['IOU'][idx] = iou
      improvedIoU += 1.0
    wellLocalizedObject = 0.0
    if iou >= minAcceptableIoU:
      wellLocalizedObject += 1.0
      if actionChosen == bss.PLACE_LANDMARK:
        wellLocalizedObject += 1.0
    visibleObject = 0.0
    if iou == 0.0:
      visibleObject = -2.0
    return improvedIoU + wellLocalizedObject + visibleObject

  def computeObjectReward(self, box, actionChosen, visitedBefore, update=True):
    iou, idx = self.matchBoxes(box)
    if actionChosen == bss.PLACE_LANDMARK:
      if iou >= 0.8: #minAcceptableIoU:
        return 3.0
      else:
        return -3.0
    else:
      improvedIoU = 0.0
      if iou > self.control['IOU'][idx]:
        if update: self.control['IOU'][idx] = iou
        improvedIoU += 1.0
      else:
        improvedIoU -= 1.0
      wellLocalizedObject = 0.0
      if iou >= minAcceptableIoU:
        wellLocalizedObject += 1.0
      visibleObject = 0.0
      if iou == 0.0:
        visibleObject = -2.0
      return improvedIoU + wellLocalizedObject + visibleObject

  def computeObjectRewardV1(self, box, actionChosen, visitedBefore, update=True):
    iou, idx = self.matchBoxes(box)
    dis, dId = self.matchCenters(box)
    dif, aId = self.matchAreas(box)
    # Agent is very close to a ground truth object
    if iou >= minAcceptableIoU:
      # Landmarks are very welcome for well localized objects
      if actionChosen == bss.PLACE_LANDMARK: 
        # We won't give double reward for the same box
        if visitedBefore:
          return -1.0
        else:
          return 3.0
      else:
        # IoU has improved for this object?
        if iou > self.control['IOU'][idx]:
          if update: self.control['IOU'][idx] = iou
          return 2.0
        else:
          return 0.0
    # Agent is not close to an object
    else:
      if iou > self.control['IOU'][idx] and update: self.control['IOU'][idx] = iou
      # Landmarks should not be placed far away from objects
      if actionChosen == bss.PLACE_LANDMARK:
        return -1.0
      else:
        iouToCenteredObject = det.IoU(box, self.boxes[dId])
        if iouToCenteredObject > 0:
          # Center is moving towards an object's center
          if dis < self.control['DIST'][dId]:
            if update: self.control['DIST'][dId] = dis
            return 1.0
        iouToCoveredObject = det.IoU(box, self.boxes[aId])
        if iouToCoveredObject > 0:
          # Area of the proposal is similar to an object's area (similar scale)
          if dif < self.control['ADIFF'][aId] and det.IoU(box, self.boxes[aId]) > 0:
            if update: self.control['ADIFF'][aId] = dif
            return 1.0
        if iouToCenteredObject > 0 or iouToCoveredObject > 0:
          return 0.0
        else:
          return -1.0

  def performAction(self, action):
    Task.performAction(self, action)

  def loadGroundTruth(self, imageName):
    try:
      gt = self.groundTruth[imageName]
    except:
      gt = []
    if self.image != imageName:
      self.displayEpisodePerformance()
      self.image = imageName
      self.boxes = [b[:] for b in gt]
      self.centers = [center(b) for b in gt]
      self.areas = [det.area(b) for b in gt]
      self.control = {'IOU': [0.0 for b in gt], 
                      'DIST': [float('inf') for b in gt], 
                      'ADIFF': [float('inf') for b in gt]}
    return gt

  def matchBoxes(self, box):
    maxIoU = -1.
    maxIdx = 0
    for i in range(len(self.boxes)):
      iou = det.IoU(box, self.boxes[i])
      if iou > maxIoU:
        maxIoU = iou
        maxIdx = i
    return (maxIoU, maxIdx)

  def matchCenters(self, box):
    minDist = float('inf')
    minIdx = 0
    c = center(box)
    for i in range(len(self.centers)):
      dist = euclideanDist(c, self.centers[i])
      if dist < minDist:
        minDist = dist
        minIdx = i
    return (minDist, minIdx)
 
  def matchAreas(self, box):
    minDiff = float('inf')
    minIdx = 0
    a = det.area(box)
    for i in range(len(self.areas)):
      diff = abs(a - self.areas[i])
      if diff < minDiff:
        minDiff = diff
        minIdx = i
    return (minDiff, minIdx)

  def displayEpisodePerformance(self):
    if self.image != '':
      detected = len( [1 for i in range(len(self.boxes)) if self.control['IOU'][i] >= 0.5] )
      recall = (float(detected)/len(self.boxes))
      maxIoU = max(self.control['IOU'])
      print self.image,'Objects detected [min(IoU) > 0.5]:',detected,'of',len(self.boxes),
      print 'Recall:',recall,
      print 'Maximum IoU achieved:',maxIoU
      self.epochRecall.append( [detected, len(self.boxes)] )
      self.epochMaxIoU.append(maxIoU)

  def flushStats(self):
    detected,expected = 0,0
    for pair in self.epochRecall:
      detected += pair[0]
      expected += pair[1]
    avgMaxIoU = np.average(self.epochMaxIoU)
    print 'Epoch Recall:',float(detected)/expected
    print 'Epoch Average MaxIoU:',avgMaxIoU
    self.epochRecall = []
    self.epochMaxIoU = []

