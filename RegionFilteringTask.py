__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task

import utils as cu
import libDetection as det

class RegionFilteringTask(Task):

  minAcceptableIoU = 0.7
  maxRejectableIoU = 0.0

  def __init__(self, environment, groundTruthFile):
    Task.__init__(self, environment)
    self.groundTruth = cu.loadBoxIndexFile(groundTruthFile)

  def getReward(self):
    img, state = self.env.getSensors()
    gt = self.getGroundTruth(img)
    rewards = []
    for s in state:
      r = self.computeReward(gt, s)
      rewards.append(r)
    self.env.updatePostReward()
    #actions = [x.lastAction for x in state]
    #print 'MDPObjectLocalizerTask::getReward(',img, actions, rewards,')'
    return rewards

  def computeReward(self, gt, sensor):
    #FIRST: Make sure sensor includes bounding boxes of action!!
    #SECOND: Go throuhg each bounding box and figure out if one of them covers an object
    #THIRD: Move on faster
    if sensor.lastAction > 1:
      # Localizing object with current image-box
      maxIoU_0, idx_0 = self.matchBoxes(sensor.prevBox, gt)
      maxIoU_1, idx_1 = self.matchBoxes(sensor.nextBox, gt)
      #if idx_0 != idx_1: print 'Focused object has changed'
      if maxIoU_1 >= 0 and maxIoU_0 >= 0:
        diff = maxIoU_1 - maxIoU_0
        if diff > 0:
          return 1.0
        else:
          return -1.0
      else:
        return -1.0
    elif sensor.lastAction == 0:
      # Previous image-box pair has been accepted
      maxIoU, idx = self.matchBoxes(sensor.prevBox, gt)
      if maxIoU >= self.minAcceptableIoU:
        return 5.0
      else:
        return -5.0
    else:
      # Previous image-box pair has been rejected
      maxIoU, idx = self.matchBoxes(sensor.prevBox, gt)
      if maxIoU <= self.maxRejectableIoU:
        return 3.0
      else:
        return -3.0

  def performAction(self, action):
    #print 'MDPObjectLocalizerTask::performAction(',action,')'
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
 
