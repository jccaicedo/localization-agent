__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task

import utils as cu
import libDetection as det

class RegionFilteringTask(Task):

  minAcceptableIoU = 0.5

  def __init__(self, environment, groundTruthFile):
    Task.__init__(self, environment)
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)

  def getReward(self):
    gt = self.getGroundTruth(self.env.db.image)
    boxes = self.env.db.boxes[self.env.state.selectedIds].tolist()
    reward = self.computeReward(gt, boxes)
    self.env.updatePostReward(reward)
    return reward

  def computeReward(self, gt, state):
    if len(state) == 0:
      return -1
    iouScores = []
    for b in state:
      iou, idx = self.matchBoxes(b, gt)
      iouScores.append(iou)
    goodBoxes = len([s for s in iouScores if s > self.minAcceptableIoU])
    maxIoU = max(iouScores)
    if goodBoxes > 0:
      return goodBoxes
    elif maxIoU > 0:
      return maxIoU
    else:
      return -0.1

  def performAction(self, action):
    Task.performAction(self, action)

  def getGroundTruth(self, imageName):
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
 
