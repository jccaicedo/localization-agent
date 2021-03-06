__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import PIL.ImageDraw as ImageDraw,PIL.Image as Image, PIL.ImageShow as ImageShow 
import os,sys
from multiprocessing import Process, JoinableQueue, Queue

import time
import utils as cu
import libDetection as det
import numpy as np

# LAYOUT CONFIGURATION
SCALES = 10
HORIZONTAL_BINS = 3
VERTICAL_BINS = 3
NUM_ACTIONS = 7
NUM_BOXES = 3 
WORLD_SIZE = SCALES*HORIZONTAL_BINS*VERTICAL_BINS
PLANE_SIZE = HORIZONTAL_BINS*VERTICAL_BINS

# ACTIONS
GO_UP    = 0
GO_DOWN  = 1
GO_FRONT = 2
GO_BACK  = 3
GO_LEFT  = 4
GO_RIGHT = 5
STAY     = 6

# OTHERS
NEAREST_NEIGHBORS = 10 # Maximum IoU

class Box():

  def __init__(self, box, id):
    self.id = id
    self.box = box
    self.area = det.area(box)

  # Symmetric Jaccard coefficient
  def IoU(self, other):
    bi = det.intersect(self.box,other.box)
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
      ua = self.area + other.area - iw*ih
      overlap = iw*ih/ua
      return overlap
    else:
      return 0

  def center(self):
    xc = self.box[0] + (self.box[2] - self.box[0])/2
    yc = self.box[1] + (self.box[3] - self.box[1])/2
    return (xc,yc)

class GraphBasedLayoutHandler():

  def __init__(self, boxes):
    t = cu.tic()
    self.auxBoxes = []
    frame = [999,999,0,0]
    id = 0
    for box in boxes.tolist():
      frame = min(frame[0:2], box[0:2]) + max(frame[2:],box[2:])
      self.auxBoxes.append(Box(box, id))
      id += 1
    self.frame = map(int,frame)
    self.auxBoxes.sort(key=lambda x:x.area, reverse=True)
    self.adjacency = np.zeros( (len(self.auxBoxes),len(self.auxBoxes)) )
    for i in range(len(self.auxBoxes)):
      self.adjacency[i,i] = 1.0
      for j in range(i+1,len(self.auxBoxes)):
        iou = self.auxBoxes[i].IoU( self.auxBoxes[j] )
        self.adjacency[i,j] = iou
        self.adjacency[j,i] = iou
    
    knn = range(-2,-GRAPH_NEIGHBORS-2,-1) # Avoid last element (same box)
    self.graph = {'nodes':[], 'edges':[]}
    for i in range(len(self.auxBoxes)):
      center = self.auxBoxes[i].center()
      node = {'data':{'id':str(i), 'box':self.auxBoxes[i].box}, 
               'position':{'x':int(center[0]),'y':int(center[1])}}
      self.graph['nodes'].append(node)
      edges = {}
      neighbors = np.argsort( self.adjacency[i,:] )
      knn =  [ (j,self.adjacency[i,j]) for j in neighbors[-NEAREST_NEIGHBORS-2:-2] ]
      for j,iou in knn:
        edge = { 'data': { 'id': str(i)+'_'+str(j), 'weight': iou, 'source': str(i), 'target': str(j) } }
        self.graph['edges'].append(edge)
    
    t = cu.toc('Graph construction:',t)
    return 

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
    self.scale = SCALES/2 # (Greedy: Set to zero)
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
    if self.actionChosen == GO_UP:      self.move( max(self.scale-1, 0), self.horizontal, self.vertical )
    elif self.actionChosen == GO_DOWN:  self.move( min(self.scale+1, SCALES-1), self.horizontal, self.vertical )
    elif self.actionChosen == GO_FRONT: self.move( self.scale, self.horizontal, max(self.vertical-1, 0) )
    elif self.actionChosen == GO_BACK:  self.move( self.scale, self.horizontal, min(self.vertical+1, VERTICAL_BINS-1))
    elif self.actionChosen == GO_LEFT:  self.move( self.scale, max(self.horizontal-1, 0), self.vertical )
    elif self.actionChosen == GO_RIGHT: self.move( self.scale, min(self.horizontal+1, HORIZONTAL_BINS-1), self.vertical )
    elif self.actionChosen == STAY:     self.move( self.scale, self.horizontal, self.vertical )
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



