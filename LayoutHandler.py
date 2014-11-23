__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import PIL.ImageDraw as ImageDraw,PIL.Image as Image, PIL.ImageShow as ImageShow 

import time
import utils as cu
import libDetection as det
import numpy as np

# LAYOUT CONFIGURATION
SCALES = 10
HORIZONTAL_BINS = 3
VERTICAL_BINS = 3
NUM_ACTIONS = 13
NUM_BOXES = 3 
WORLD_SIZE = SCALES*HORIZONTAL_BINS*VERTICAL_BINS
PLANE_SIZE = HORIZONTAL_BINS*VERTICAL_BINS

# ACTIONS
EXPLORE_ONE_SCALE_UP    = 0
EXPLORE_ONE_SCALE_DOWN  = 1
EXPLORE_TWO_SCALES_UP   = 2
EXPLORE_TWO_SCALES_DOWN = 3
GOTO_TOP_LEFT           = 4
GOTO_TOP_CENTER         = 5
GOTO_TOP_RIGHT          = 6
GOTO_MIDDLE_LEFT        = 7
GOTO_MIDDLE_CENTER      = 8
GOTO_MIDDLE_RIGHT       = 9
GOTO_BOTTOM_LEFT        = 10
GOTO_BOTTOM_CENTER      = 11
GOTO_BOTTOM_RIGHT       = 12

class Box():

  def __init__(self, box, id):
    self.id = id
    self.box = box
    self.area = det.area(box)

class LayoutHandler():

  def __init__(self, boxes):
    auxBoxes = []
    frame = [999,999,0,0]
    id = 0
    for box in boxes.tolist():
      frame = min(frame[0:2], box[0:2]) + max(frame[2:],box[2:])
      auxBoxes.append(Box(box, id))
      id += 1
    self.frame = map(int,frame)
    auxBoxes.sort(key=lambda x:x.area, reverse=True)
    self.layout = []
    scaleRange = len(auxBoxes)/SCALES
    for s in range(SCALES):
      scaleElems = []
      self.layout.append([])
      for i in range(scaleRange):
        scaleElems.append(auxBoxes[scaleRange*s + i])
      scaleElems.sort(key=lambda x:x.box[0], reverse=True)
      horizontalRange = len(scaleElems)/HORIZONTAL_BINS
      for h in range(HORIZONTAL_BINS):
        horizontalRangeElems = []
        self.layout[s].append([])
        for j in range(horizontalRange):
          horizontalRangeElems.append(scaleElems[horizontalRange*h + j])
        horizontalRangeElems.sort(key=lambda x:x.box[1], reverse=True)
        verticalRange = len(horizontalRangeElems)/VERTICAL_BINS
        for v in range(VERTICAL_BINS):
          self.layout[s][h].append([])
          for k in range(verticalRange):
            self.layout[s][h][v].append(horizontalRangeElems[verticalRange*v + k])
    self.numBoxes = len(auxBoxes)
    self.boxesPerBin = float(self.numBoxes)/WORLD_SIZE
    self.actionCounter = 0
    self.scale = 0 #SCALES/2 # (Greedy: Set to zero)
    self.horizontal = 0
    self.vertical = 0
    self.percentExplored = 0
    self.selectedIds = []
    self.status = np.zeros( (SCALES, HORIZONTAL_BINS, VERTICAL_BINS), dtype=np.int32 )
    self.currentPosition = np.zeros( (SCALES, HORIZONTAL_BINS, VERTICAL_BINS), dtype=np.int32 )

  def move(self, s, h, v):
    self.scale = s
    self.horizontal = h
    self.vertical = v

  def performAction(self, action):
    if self.percentExplored == 1.0:
      return 'Done'
    self.actionChosen = action[0]
    self.actionValue = action[1]
    self.actionCounter += 1
    if self.actionChosen == EXPLORE_ONE_SCALE_UP:      self.move( max(self.scale-1, 0), self.horizontal, self.vertical )
    elif self.actionChosen == EXPLORE_ONE_SCALE_DOWN:  self.move( min(self.scale+1, SCALES-1), self.horizontal, self.vertical )
    elif self.actionChosen == EXPLORE_TWO_SCALES_UP:   self.move( max(self.scale-2, 0), self.horizontal, self.vertical )
    elif self.actionChosen == EXPLORE_TWO_SCALES_DOWN: self.move( min(self.scale+2, SCALES-1), self.horizontal, self.vertical )
    elif self.actionChosen == GOTO_TOP_LEFT:           self.move( self.scale, 0, 0 )
    elif self.actionChosen == GOTO_TOP_CENTER:         self.move( self.scale, 1, 0 )
    elif self.actionChosen == GOTO_TOP_RIGHT:          self.move( self.scale, 2, 0 )
    elif self.actionChosen == GOTO_MIDDLE_LEFT:        self.move( self.scale, 0, 1 )
    elif self.actionChosen == GOTO_MIDDLE_CENTER:      self.move( self.scale, 1, 1 )
    elif self.actionChosen == GOTO_MIDDLE_RIGHT:       self.move( self.scale, 2, 1 )
    elif self.actionChosen == GOTO_BOTTOM_LEFT:        self.move( self.scale, 0, 2 )
    elif self.actionChosen == GOTO_BOTTOM_CENTER:      self.move( self.scale, 1, 2 )
    elif self.actionChosen == GOTO_BOTTOM_RIGHT:       self.move( self.scale, 2, 2 )
    return self.selectBoxes()

  def selectBoxes(self):
    self.selectedIds = []
    self.currentPosition[:,:,:] = 0
    # 2D Plane exploration
    ini = self.status[self.scale,self.horizontal,self.vertical]
    end = min( ini + NUM_BOXES, len(self.layout[self.scale][self.horizontal][self.vertical]) )
    for i in range(ini, end):
      self.selectedIds.append(self.layout[self.scale][self.horizontal][self.vertical][i].id)
    self.status[self.scale, self.horizontal, self.vertical] = end
    self.currentPosition[self.scale, self.horizontal, self.vertical] = 1.0
    return self.selectedIds

  def getLocationState(self):
    state = np.copy( self.status[self.scale, ...].reshape( (PLANE_SIZE) ) )
    state = state / self.boxesPerBin
    location = np.zeros( (SCALES + PLANE_SIZE) )
    location[self.scale] = 1.0
    location[SCALES:] = self.currentPosition[self.scale, ...].reshape( (PLANE_SIZE) )
    return np.concatenate( (state, 3*location) )

  def isNewExploredCell(self):
    return self.status[self.scale, self.horizontal, self.vertical] <= NUM_BOXES


