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

  def __init__(self, environment, groundTruthFile):
    Task.__init__(self, environment)
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)
    self.image = ''
    self.prevPos = (0,0,0)

  def getReward(self):
    gt = self.loadGroundTruth(self.env.imageList[self.env.idx])
    reward = self.computeObjectReward(self.env.state)
    self.env.updatePostReward(reward)
    return reward

  def computeObjectReward(self, state):
    iou, idx = self.matchBoxes(state.box)
    dis, dId = self.matchCenters(state.box)
    dif, aId = self.matchAreas(state.box)
    # Agent is very close to a ground truth object
    if iou >= minAcceptableIoU:
      # Landmarks are very welcome for well localized objects
      if state.actionChosen == bss.PLACE_LANDMARK and not state.visitedBefore():
        return 3.0
      else:
        # IoU has improved for this object?
        if iou >= self.control['IOU'][idx]:
          self.control['IOU'][idx] = iou
          return 3.0
        else:
          return 0.0
    # Agent is not close to an object
    else:
      # Landmarks should not be placed far away from objects
      if state.actionChosen == bss.PLACE_LANDMARK:
        return -1.0
      else:
        # Center is moving towards an object's center
        if dis < self.control['DIST'][dId] and det.IoU(state.box, self.boxes[dId]) > 0:
          self.control['DIST'][dId] = dis
          return 1.0
        else:
          # Area of the proposal is similar to an object's area (similar scale)
          if dif < self.control['ADIFF'][aId] and det.IoU(state.box, self.boxes[aId]) > 0:
            self.control['ADIFF'][aId] = dif
            return 1.0
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

