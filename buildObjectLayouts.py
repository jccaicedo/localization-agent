import os,sys
import utils as cu
import libDetection as det
import numpy as np

MAX_NUMBER_OF_PARTS = 5

def dist(x,y):
  d = 0
  for i in range(len(x)):
    d += (x[i]-y[i])**2
  return d

def S(t):
  return 1.0/(1.0 + np.exp(-t))

class ObjectLayout():

  def __init__(self, img):
    self.imageName = img
    self.root = []
    self.context = []
    self.parts = []

  def addRootAndContext(self, root, context, sharedAreas):
    self.root = root
    self.context = context
    self.IoURootContext = sharedAreas[0]
    self.CoverContextRoot = sharedAreas[1]

  def addPart(self, part, sharedAreas):
    self.parts.append(part+sharedAreas)

  def getScore(self):
    return self.getProductOfScores()

  def getProductOfScores(self):
    score = 1
    if len(self.root) > 0:
      score *= S(self.root[0])
    if len(self.context) > 0:
      score *= S(self.context[0])
    if len(self.parts) > 0:
      parts = 1
      for p in self.parts:
        parts *= S(p[0])
    else:
      parts = -1
    return score + parts
   
  def getSumOfScores(self):
    score = 0
    if len(self.root) > 0:
      score += self.root[0]
    if len(self.context) > 0:
      score += self.context[0]
    if len(self.parts) > 0:
      for p in self.parts:
        score += p[0]
    return score

  def getLayoutData(self):
    boxes = []
    scores = []
    if len(self.root) > 0:
      boxes.append(self.root[1:])
      scores.append(self.root[0])
    if len(self.context) > 0:
      boxes.append(self.context[1:])
      scores.append(self.context[0])
    if len(self.parts) > 0:
      for p in self.parts:
        boxes.append(p[1:])
        scores.append(p[0])
    return self.imageName, boxes, scores

  def printStructure(self):
    print self.imageName,
    print 'Context-Root [IoU:','{:4.2f}'.format(self.IoURootContext),
    print 'Cov:','{:4.2f}'.format(self.CoverContextRoot),']',
    print 'Root:','{:5.3f}'.format(self.root[0]),'Context:','{:5.3f}'.format(self.context[0]),
    print 'Parts:',
    for p in self.parts:
      print '(','{:4.2f}'.format(p[-2]),'{:4.2f}'.format(p[-1]),'{:5.3f}'.format(p[0]),')',
    print 'Total Score:',self.getScore()

def findBigMatches(img, big, tight):
  layouts = []
  idealMatch = [0.25, 1.0]
  for i in range(len(big)):
    b = big[i]
    for j in range(len(tight)):
      t = tight[j]
      r = [ det.IoU(b[1:], t[1:]), det.overlap(b[1:], t[1:])]
      if 0.8 >= r[0] and r[1] >= 0.8:
        s = b[0] + t[0]
        ol = ObjectLayout(img)
        ol.addRootAndContext(t,b,r)
        layouts.append(ol)
  layouts.sort(key=lambda x: x.getScore(), reverse=True)
  return layouts

def findInsideMatches(inside, layouts):
  idealMatch = [0.25, 1.0]
  for i in range(len(layouts)):
    t = layouts[i].root
    candidates = []
    for j in range(len(inside)):
      n = inside[j]
      r = [ det.IoU(n[1:], t[1:]), det.overlap(t[1:], n[1:]) ]
      if 0.8 >= r[0] and r[1] >= 0.8:
        s = np.exp( -dist(idealMatch, r) )
        candidates.append( [j,s,n[0],r] )
    if len(candidates) > 0:
      candidates.sort(key=lambda x:x[1]+x[2],reverse=True)
      for k in range( min(MAX_NUMBER_OF_PARTS,len(candidates)) ):
        layouts[i].addPart( inside[ candidates[k][0] ], candidates[k][-1] )
  layouts.sort(key=lambda x: x.getScore(), reverse=True)
  return layouts

params = cu.loadParams('bigDetections tightDetections insideDetections imageDir outputDir')

big = cu.loadBoxIndexFile( params['bigDetections'] )
tight = cu.loadBoxIndexFile( params['tightDetections'] )
inside = cu.loadBoxIndexFile( params['insideDetections'] )

print 'Images:',len(big),len(tight),len(inside)

allLayouts = []
for k in big.keys():
  layouts = findBigMatches(k, big[k], tight[k])
  layouts = findInsideMatches(inside[k], layouts)
  allLayouts += [ layouts[0] ]

allLayouts.sort(key=lambda x: x.getScore(), reverse=True)
matchCounter = 0
for ol in allLayouts:
  img, boxes, scores = ol.getLayoutData()
  ol.printStructure()
  det.showObjectLayout(params['imageDir'] + img + '.jpg', boxes, scores, outputFile=params['outputDir'] + str(matchCounter) + '.png')
  matchCounter += 1
  if matchCounter >= 50: break

