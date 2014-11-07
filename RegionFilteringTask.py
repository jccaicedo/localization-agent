__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task

import utils as cu
import libDetection as det
import numpy as np

class RegionFilteringTask(Task):

  minAcceptableIoU = 0.5

  def __init__(self, environment, groundTruthFile):
    Task.__init__(self, environment)
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)
    self.image = ''
    self.prevAction = 0
    self.prevPos = (0,0,0)

  def getReward(self):
    gt = self.getGroundTruth(self.env.db.image)
    boxes = self.env.db.boxes[self.env.state.selectedIds].tolist()
    objectDiscoveryReward = self.computeObjectReward(gt, boxes)
    explorationReward = self.computeNavigationReward(self.env.state)
    reward = objectDiscoveryReward + explorationReward
    self.env.updatePostReward(reward)
    self.prevAction = self.env.state.actionChosen
    return reward

  def computeObjectReward(self, gt, state):
    if len(state) == 0:
      # Visiting a region with no boxes
      return -1.0
    return 0.0 # Do not count boxes overlap
    iouScores = []
    for b in state:
      iou, idx = self.matchBoxes(b, gt)
      iouScores.append(iou)
    goodBoxes = len([s for s in iouScores if s > self.minAcceptableIoU])
    maxIoU = max(iouScores)
    if goodBoxes > 0:
      return 5.0
    elif maxIoU > 0.2:
      return 1.0
    else:
      return 0.0

  def computeNavigationReward(self, state):
    navigationReward = 0.0
    # Visiting a region that has not been explored yet
    if state.isNewExploredCell():
      navigationReward += 0.51
    # Moving to a different position
    if self.prevPos[0] != state.scale or self.prevPos[1] != state.horizontal or self.prevPos[2] != state.vertical:
      navigationReward += 0.52
    self.prevPos = state.scale,state.horizontal,state.vertical
    return navigationReward

  def performAction(self, action):
    Task.performAction(self, action)

  def getGroundTruth(self, imageName):
    if self.image != imageName:
      self.prevPos = (0,0,0)
      self.prevAction = 0
      self.image = imageName
    try:
      gt = self.groundTruth[imageName]
    except:
      gt = []
    return gt

  def matchBoxes(self, box, gt):
    maxIoU = -1.
    maxIdx = 0
    for i in range(len(gt)):
      iou = det.IoU(box, gt[i])
      if iou > maxIoU:
        maxIoU = iou
        maxIdx = i
    return (maxIoU, maxIdx)
 
