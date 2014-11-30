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
    self.epochLandmarks = []

  def getReward(self):
    gt = self.loadGroundTruth(self.env.imageList[self.env.idx])
    reward = self.computeObjectReward(self.env.state.box, self.env.state.actionChosen)
    allDone = reduce(lambda x,y: x and y, self.control['DONE'])
    self.env.updatePostReward(reward, allDone, self.cover)
    return reward

  def computeObjectReward(self, box, actionChosen, update=True):
    reward = 0
    self.cover = False
    iou, idx = self.matchBoxes(box)
    if iou == 0.0:
      reward = -2.0
    else:
      improvedIoU = False
      if iou > self.control['IOU'][idx]:
        if update: self.control['IOU'][idx] = iou
        improvedIoU = True
      if not improvedIoU and iou < 0.5:
        reward = -1.0
      elif improvedIoU and iou < 0.5:
        reward = 1.0
      elif not improvedIoU and iou >= 0.5 and actionChosen != bss.PLACE_LANDMARK:
        reward = -1.0
      elif improvedIoU and iou >= 0.5:
        reward = 2.0
      elif actionChosen == bss.PLACE_LANDMARK:
        if iou >= minAcceptableIoU:
          if update: 
            self.control['DONE'][idx] = True
            self.cover = True
            for j in range(len(self.control['IOU'])):
              if not self.control['DONE']:
                self.control['IOU'] = 0.0
        if iou < 0.5:
          reward = -1.0
        elif iou < 0.7:
          reward = 2.0
        else:
          reward = 3.0
    return reward

  def computeObjectRewardV5(self, box, actionChosen, update=True):
    iou, idx = self.matchBoxes(box)
    if actionChosen == bss.PLACE_LANDMARK:
      if iou >= 0.7: #minAcceptableIoU:
        if update: 
          self.control['DONE'][idx] = True
          for j in range(len(self.control['IOU'])):
            if not self.control['DONE']:
              self.control['IOU'] = 0.0
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
      self.control = {'IOU': [0.0 for b in gt], 'DONE': [False for b in gt]} 
    return gt

  def matchBoxes(self, box):
    maxIoU = -1.
    maxIdx = 0
    for i in range(len(self.boxes)):
      if self.control['DONE'][i]: 
        continue
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
      landmarks = sum( self.control['DONE'] )
      print self.image,'Objects detected [min(IoU) > 0.5]:',detected,'of',len(self.boxes),
      print 'Recall:',recall,
      print 'Maximum IoU achieved:',maxIoU,
      print 'Landmarks:',landmarks
      self.epochRecall.append( [detected, len(self.boxes)] )
      self.epochMaxIoU.append(maxIoU)
      self.epochLandmarks.append(landmarks)

  def flushStats(self):
    detected,expected = 0,0
    for pair in self.epochRecall:
      detected += pair[0]
      expected += pair[1]
    landmarks = sum(self.epochLandmarks)
    avgMaxIoU = np.average(self.epochMaxIoU)
    print 'Epoch Recall:',float(detected)/expected
    print 'Epoch Average MaxIoU:',avgMaxIoU
    print 'Epoch Landmarks:',float(landmarks)/expected
    self.epochRecall = []
    self.epochMaxIoU = []
    self.epochLandmarks = []

