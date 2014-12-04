__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.environments import Task
import BoxSearchState as bss

import utils as cu
import libDetection as det
import numpy as np

import RLConfig as config

MIN_ACCEPTABLE_IOU = config.geti('minAcceptableIoU')

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
    self.cover = []
    iou, idx = self.matchBoxes(box)
    if iou <= 0.0:
      reward = -2.0
    else:
      improvedIoU = False
      if iou > self.control['IOU'][idx]:
        if update: self.control['IOU'][idx] = iou
        improvedIoU = True
      if not improvedIoU and actionChosen != bss.PLACE_LANDMARK:
        reward = -1.0
      elif improvedIoU and iou < 0.5:
        reward = 1.0
      elif improvedIoU and iou >= 0.5:
        reward = 2.0
      elif actionChosen == bss.PLACE_LANDMARK:
        if iou >= MIN_ACCEPTABLE_IOU:
          if update: 
            self.coverSample(idx)
            for j in range(len(self.control['IOU'])):
              if not self.control['DONE'][j]:
                self.control['IOU'][j] = 0.0
        if iou < MIN_ACCEPTABLE_IOU:
          reward = -1.0
        else:
          reward = 3.0
    return reward

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
      self.control = {'IOU': [0.0 for b in gt], 'DONE': [False for b in gt], 
                      'SKIP': [False for b in gt]} 
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

  def coverSample(self, idx):
    self.control['DONE'][idx] = True
    self.cover = self.boxes[idx][:]
    conflicts = []
    for j in range(len(self.control['DONE'])):
      if not self.control['DONE'][j]:
        ov = det.overlap(self.cover, self.boxes[j])
        if ov > 0.0:
          conflicts.append( (j,ov) )
    conflicts.sort(key=lambda x:x[1], reverse=True)
    for j,ov in conflicts:
      nov = det.overlap(self.cover, self.boxes[j])
      if nov < 0.5:
        ib = det.intersect(self.cover, self.boxes[j])
        iw = ib[2] - ib[0] + 1
        ih = ib[3] - ib[1] + 1
        if iw > ih: # Cut height first
          if self.cover[1] >= ib[1]: self.cover[1] = ib[3]
          if self.cover[3] <= ib[3]: self.cover[3] = ib[1]
        else: # Cut width first
          if self.cover[0] >= ib[0]: self.cover[0] = ib[2]
          if self.cover[2] <= ib[2]: self.cover[2] = ib[0]
      elif nov >= 0.5: # Assume the example is done for now
        self.control['DONE'][j] = True
        self.control['SKIP'][j] = True

  def displayEpisodePerformance(self):
    if self.image != '':
      detected = len( [1 for i in range(len(self.boxes)) if self.control['IOU'][i] >= 0.5 and not self.control['SKIP'][i]] )
      recall = (float(detected)/len(self.boxes))
      maxIoU = max(self.control['IOU'])
      landmarks = sum( self.control['DONE'] ) - sum( self.control['SKIP'] )
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

