__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task

import utils as cu
import libDetection as det
import numpy as np

minAcceptableIoU = 0.5

class BoxSearchTask(Task):

  def __init__(self, environment, groundTruthFile):
    Task.__init__(self, environment)
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)
    self.image = ''
    self.prevPos = (0,0,0)

  def getReward(self):
    gt = self.getGroundTruth(self.env.imageList[self.env.idx])
    reward = self.computeObjectReward(gt, self.env.state)
    self.env.updatePostReward(reward)
    return reward

  def computeObjectReward(self, gt, state):
    if state.visitedBefore():
      return -1.0
    iou, idx = self.matchBoxes(state.box, gt)
    if iou > 0.2:
      if iou > self.maxIoU[idx]:
        # Agent is improving IoU of this object. The reward could be scaled by a constant factor
        self.maxIoU[idx] = iou
        return iou
      else:
        # IoU is not improving. We could give negative rewards to repell the agent from this position
        return -0.1
    else:
      return 0.0
      
  def performAction(self, action):
    Task.performAction(self, action)

  def getGroundTruth(self, imageName):
    try:
      gt = self.groundTruth[imageName]
    except:
      gt = []
    if self.image != imageName:
      self.image = imageName
      self.maxIoU = [0.0 for b in gt]
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
 
