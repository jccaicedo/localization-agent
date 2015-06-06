__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task
import TrackerState as ts

import utils.utils as cu
import utils.libDetection as det
import numpy as np

import learn.rl.RLConfig as config

MIN_ACCEPTABLE_IOU = config.getf('minAcceptableIoU')

def center(box):
  return [ (box[2] + box[0])/2.0 , (box[3] + box[1])/2.0 ]

def euclideanDist(c1, c2):
  return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2

class TrackerTask(Task):

  def __init__(self, environment=None):
    Task.__init__(self, environment)
    self.image = ''
    self.epochRecall = []
    self.epochMaxIoU = []
    self.epochLandmarks = []

  def getReward(self):
    self.loadGroundTruth(self.env.currentFrameName(), self.env.currentFrameGroundTruth())
    reward = self.computeObjectReward(self.env.state.box, self.env.state.actionChosen)
    self.env.updatePostReward(reward)
    return reward

  def computeObjectReward(self, box, actionChosen, update=True):
    reward = 0
    iou, idx = self.matchBoxes(box)
    if iou <= 0.0:
      reward = -2.0
    else:
      improvedIoU = False
      if iou > self.control['IOU'][idx]:
        if update: self.control['IOU'][idx] = iou
        improvedIoU = True
      if not improvedIoU and actionChosen != ts.PLACE_LANDMARK:
        reward = -1.0
      elif improvedIoU and iou < 0.7:
        reward = 1.0
      elif improvedIoU and iou >= 0.7:
        reward = 2.0
      elif actionChosen == ts.PLACE_LANDMARK:
        if iou >= MIN_ACCEPTABLE_IOU:
          self.control['DONE'][idx] = True
          if update: 
            for j in range(len(self.control['IOU'])):
              if not self.control['DONE'][j]:
                self.control['IOU'][j] = 0.0
        if iou < 0.7:
          reward = -1.0
        if iou < MIN_ACCEPTABLE_IOU and iou > 0.7:
          reward = 1.0
        else:
          reward = 3.0
    return reward

  def performAction(self, action):
    Task.performAction(self, action)

  def loadGroundTruth(self, imageName, gt):
    if self.image != imageName:
      #self.displayEpisodePerformance()
      self.image = imageName
      self.boxes = [b[:] for b in gt]
      self.centers = [center(b) for b in gt]
      self.areas = [det.area(b) for b in gt]
      self.control = {'IOU': [0.0 for b in gt], 'DONE': [False for b in gt]} 

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

