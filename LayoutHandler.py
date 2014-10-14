__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import PIL.ImageDraw as ImageDraw,PIL.Image as Image, PIL.ImageShow as ImageShow 

import time
import utils as cu
import libDetection as det

MAX_NUM_BOXES = 10

class LayoutHandler():

  # ACTIONS
  ACCEPT = 0
  REJECT = 1
  EXPAND_VERY_FAR = 2
  EXPAND_FAR = 3
  EXPAND_CLOSE = 4
  EXPAND_VERY_CLOSE = 5
  SPLIT_VERY_FAR = 6
  SPLIT_FAR = 7
  SPLIT_CLOSE = 8
  SPLIT_VERY_CLOSE = 9

  def __init__(self, boxes):
    self.boxes = []
    self.areas = []
    self.status = []
    frame = [999,999,0,0]
    for b in boxes:
      box = map(float, b[1:])
      frame = min(frame[0:2], box[0:2]) + max(frame[2:],box[2:])
      self.boxes.append(box)
      self.areas.append(det.area(box))
    self.boxes.append(frame)
    self.areas.append(det.area(frame))
    self.selectBox( len(self.boxes)-1 )
    self.frame = map(int,frame)

  def performAction(self, action):
    if   action == self.ACCEPT:             return None
    elif action == self.REJECT:             return None
    elif action == self.EXPAND_VERY_FAR:    return self.expand(0.1, 0.3)
    elif action == self.EXPAND_FAR:         return self.expand(0.3, 0.5)
    elif action == self.EXPAND_CLOSE:       return self.expand(0.5, 0.7)
    elif action == self.EXPAND_VERY_CLOSE:  return self.expand(0.7, 0.9)
    elif action == self.SPLIT_VERY_FAR:     return self.split(0.1, 0.3)
    elif action == self.SPLIT_FAR:          return self.split(0.3, 0.5)
    elif action == self.SPLIT_CLOSE:        return self.split(0.5, 0.7)
    elif action == self.SPLIT_VERY_CLOSE:   return self.split(0.7, 0.9)

  def selectBox(self, boxIdx):
    self.selectedBox = boxIdx
    self.computeOverlaps()

  def computeOverlaps(self):
    self.intersection = []
    self.iou = []
    self.intersectedIdx = []
    b1 = self.boxes[self.selectedBox]
    for i in range(len(self.boxes)):
      b2 = self.boxes[i]
      bi = det.intersect(b1,b2)
      iw = bi[2] - bi[0] + 1
      ih = bi[3] - bi[1] + 1
      if iw > 0 and ih > 0:
        areaIntersection = iw*ih
        areaUnion = self.areas[self.selectedBox] + self.areas[i] - areaIntersection
        iou = areaIntersection/areaUnion
        self.intersectedIdx.append(i)
      else:
        iou = 0.0
        areaIntersection = 0.0
      self.intersection.append(areaIntersection)
      self.iou.append(iou)
    self.intersectedIdx.sort(key=lambda x: self.iou[x], reverse=True)

  def split(self, minIoU, maxIoU):
    candidates = [i for i in self.intersectedIdx 
        if self.iou[i] >= minIoU and self.iou[i] <= maxIoU and self.intersection[i]/self.areas[i] >= 0.9]
    candidates.sort(key=lambda x: self.areas[x], reverse=True) # Sort by \sum(IoU) of all candidates
    self.displayCandidates(candidates)
    return candidates[0:MAX_NUM_BOXES]

  def expand(self, minIoU, maxIoU):
    candidates = [i for i in self.intersectedIdx
        if self.iou[i] >= minIoU and self.iou[i] <= maxIoU and self.intersection[i]/self.areas[self.selectedBox] >= 0.9]
    candidates.sort(key=lambda x: self.areas[x], reverse=True) # Sort by \sum(IoU) of all candidates
    self.displayCandidates(candidates)
    return candidates[0:MAX_NUM_BOXES]

  def getBoxesInfo(self, candidates):
    A,B = [],[]
    for i in candidates:
      A.append(self.areas[i]/float(self.areas[-1]))
      B.append(self.boxes[i])
    return A,B
    
  def displayCandidates(self, candidates):
    im = Image.new("RGB", self.frame[2:])
    draw = ImageDraw.Draw(im)
    draw.rectangle(self.boxes[self.selectedBox], fill=(100,100,100))
    for c in candidates[0:10]:
      draw.rectangle(self.boxes[c], outline=(255,2*c+10,10*c))
    im.show()
  

