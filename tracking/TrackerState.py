__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import time
import numpy as np
from PIL import Image
import random
import os

# ACTIONS
X_1_UP         = 0
Y_1_UP         = 1
X_2_UP         = 2
Y_2_UP         = 3
X_1_DOWN       = 4
Y_1_DOWN       = 5
X_2_DOWN       = 6
Y_2_DOWN       = 7
PLACE_LANDMARK = 8
ABORT = -1

# BOX LIMITS
MIN_BOX_SIDE     = 10
STEP_FACTOR      = 0.05
MIN_IoU          = 0.90
MAX_ACTIONS      = 50

# OTHER DEFINITIONS
NUM_ACTIONS = 9

intersect = lambda x,y: [max(x[0],y[0]),max(x[1],y[1]),min(x[2],y[2]),min(x[3],y[3])]

area = lambda x: (x[2]-x[0]+1)*(x[3]-x[1]+1)

# Symmetric Jaccard coefficient
def IoU(b1,b2):
  bi = intersect(b1,b2)
  iw = bi[2] - bi[0] + 1
  ih = bi[3] - bi[1] + 1
  if iw > 0 and ih > 0:
    ua = area(b1) + area(b2) - iw*ih
    overlap = iw*ih/ua
    return overlap
  else:
    return 0

# How much box1 covers box2
def overlap(b1,b2):
  bi = intersect(b1,b2)
  iw = bi[2] - bi[0] + 1
  ih = bi[3] - bi[1] + 1
  if iw > 0 and ih > 0:
    #ua = area(b1) + area(b2) - iw*ih
    overlap = iw*ih/float(area(b2))
    return overlap
  else:
    return 0

class TrackerState():

  def __init__(self, image, target=None):
    self.visibleImage = image
    self.box = [0,0,0,0]
    self.landmarkIndex = {}
    self.actionChosen = 2
    self.target = target
    self.reset()
    self.stepsWithoutLandmark = 0

  def performAction(self, action):
    self.actionChosen = action
    self.stepsWithoutLandmark += 1

    if action == X_1_UP:               newBox = self.x1Up()
    elif action == Y_1_UP:             newBox = self.y1Up()
    elif action == X_2_UP:             newBox = self.x2Up()
    elif action == Y_2_UP:             newBox = self.y2Up()
    elif action == X_1_DOWN:           newBox = self.x1Down()
    elif action == Y_1_DOWN:           newBox = self.y1Down()
    elif action == X_2_DOWN:           newBox = self.x2Down()
    elif action == Y_2_DOWN:           newBox = self.y2Down()
    elif action == PLACE_LANDMARK:     newBox = self.placeLandmark()

    self.box = newBox
    self.boxW = self.box[2] - self.box[0]
    self.boxH = self.box[3] - self.box[1]
    return self.box

  def x1Up(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxW
    if self.box[0] + step + MIN_BOX_SIDE < self.visibleImage.size[0]:
      newBox[0] += step
    else:
      newBox[0] = self.visibleImage.size[0] - MIN_BOX_SIDE - 1
    return self.adjustAndClip(newBox)

  def y1Up(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxH
    if self.box[1] + step + MIN_BOX_SIDE < self.visibleImage.size[1]:
      newBox[1] += step
    else:
      newBox[1] = self.visibleImage.size[1] - MIN_BOX_SIDE - 1
    return self.adjustAndClip(newBox)

  def x2Up(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxW
    if self.box[2] + step < self.visibleImage.size[0]:
      newBox[2] += step
    else:
      newBox[2] = self.visibleImage.size[0] - 1
    return self.adjustAndClip(newBox)

  def y2Up(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxH
    if self.box[3] + step < self.visibleImage.size[1]:
      newBox[3] += step
    else:
      newBox[3] = self.visibleImage.size[1] - 1
    return self.adjustAndClip(newBox)

  def x1Down(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxW
    if self.box[0] - step >= 0:
      newBox[0] -= step
    else:
      newBox[0] = 0
    return self.adjustAndClip(newBox)

  def y1Down(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxH
    if self.box[1] - step >= 0:
      newBox[1] -= step
    else:
      newBox[1] = 0
    return self.adjustAndClip(newBox)

  def x2Down(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxW
    if self.box[2] - step >= MIN_BOX_SIDE:
      newBox[2] -= step
    else:
      newBox[2] = MIN_BOX_SIDE
    return self.adjustAndClip(newBox)

  def y2Down(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxH
    if self.box[3] - step >= MIN_BOX_SIDE:
      newBox[3] -= step
    else:
      newBox[3] = MIN_BOX_SIDE
    return self.adjustAndClip(newBox)

  def xCoordUp(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxW
    if self.box[0] + step + self.boxW < self.visibleImage.size[0]:
      newBox[0] += step
      newBox[2] += step
    else:
      newBox[0] = self.visibleImage.size[0] - self.boxW - 1
      newBox[2] = self.visibleImage.size[0] - 1
    return self.adjustAndClip(newBox)

  def yCoordUp(self):
    newBox = self.box[:]
    step = STEP_FACTOR*self.boxH
    if self.box[1] + step + self.boxH < self.visibleImage.size[1]:
      newBox[1] += step
      newBox[3] += step
    else:
      newBox[1] = self.visibleImage.size[1] - self.boxH - 1
      newBox[3] = self.visibleImage.size[1] - 1
    return self.adjustAndClip(newBox)

  def adjustAndClip(self, box):
    if box[0] < 0:
      # Can we move it to the right?
      step = -box[0]
      if box[2] + step < self.visibleImage.size[0]:
        box[0] += step
        box[2] += step
      else:
        box[0] = 0
        box[2] = self.visibleImage.size[0] - 1
    if box[1] < 0:
      step = -box[1]
      # Can we move it down?
      if box[3] + step < self.visibleImage.size[1]:
        box[1] += step
        box[3] += step
      else:
        box[1] = 0
        box[3] = self.visibleImage.size[1] - 1
    if box[2] >= self.visibleImage.size[0]:
      step = box[2] - self.visibleImage.size[0]
      # Can we move it to the left?
      if box[0] - step >= 0:
        box[0] -= step
        box[2] -= step
      else:
        box[0] = 0
        box[2] = self.visibleImage.size[0] - 1
    if box[3] >= self.visibleImage.size[1]:
      step = box[3] - self.visibleImage.size[1]
      # Can we move it up?
      if box[1] - step >= 0:
        box[1] -= step
        box[3] -= step
      else:
        box[1] = 0
        box[3] = self.visibleImage.size[1] - 1
    return box

  def placeLandmark(self):
    self.stepsWithoutLandmark = 0
    return self.box

  def reset(self):
    self.stepsWithoutLandmark = 0
    self.box = [0,0,self.visibleImage.size[0],self.visibleImage.size[1]]
    self.boxW = self.box[2]-self.box[0]
    self.boxH = self.box[3]-self.box[1]
    self.aspectRatio = self.boxH/self.boxW

  def sampleBestAction(self):
    if self.target is None:
      return np.argmax( np.random.random([1, NUM_ACTIONS]), 1 )
    else:
      nextBoxes = []
      nextBoxes.append( self.x1Up() )
      nextBoxes.append( self.y1Up() )
      nextBoxes.append( self.x2Up() )
      nextBoxes.append( self.y2Up() )
      nextBoxes.append( self.x1Down() )
      nextBoxes.append( self.y1Down() )
      nextBoxes.append( self.x2Down() )
      nextBoxes.append( self.y2Down() )
      #nextBoxes.append( self.placeLandmark() )

      rewards = []
      baseline = IoU(self.target,self.box)
      if baseline >= MIN_IoU:
        return PLACE_LANDMARK
      if self.stepsWithoutLandmark >= MAX_ACTIONS:
        return ABORT
      for a in range(len(nextBoxes)):
        r = IoU(nextBoxes[a],self.target)-baseline
        rewards.append(r)
      best = np.argmax(rewards)
      if rewards[best] == 0.0:
        best = np.random.randint(0,NUM_ACTIONS-1)
      return best
      
