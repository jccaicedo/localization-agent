import os
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import TrajectorySimulator as ts
try:
  import cv2
except:
  cv2 = None

#TODO: Put this configuration in an external file or rely entirely on Coco's data
#dataDir = '/home/jccaicedo/data/tracking/simulations/'
dataDir = '/home/juan/Pictures/test/'
scene = dataDir + 'bogota.jpg'
obj = dataDir + 'photo.jpg'
box = [0, 100, 0, 100]
polygon = [50, 0, 100, 50, 50, 100, 0, 50]
imgSize = 200
channels = 2
totalFrames = 60
cam = False


def fraction(b,k):
  w = (b[2]-b[0])*(1-k)/2.
  h = (b[3]-b[1])*(1-k)/2.
  return [b[0]+w,b[1]+h, b[2]-w,b[3]-h]

def maskFrame(frame, flow, box):
  if flow is not None:
    maskedF = np.zeros( (4, frame.shape[0], frame.shape[1]) )
    maskedF[1,:,:] = flow[...,0]/10
    maskedF[2,:,:] = flow[...,1]/10
  else:
    maskedF = np.zeros( (2, frame.shape[0], frame.shape[1]) )

  maskedF[0,:,:] = (frame - 128.0)/128.0
  maskedF[-1,:,:] = 0
  for factor in [(1.00, 1), (0.75, -1), (0.5, 1), (0.25, -1)]:
    b = map(int, fraction(box, factor[0]))
    maskedF[-1, b[0]:b[2], b[1]:b[3]] = factor[1]
  #import pylab
  #pylab.imshow(maskedF[-1,:,:])
  #pylab.show()
  return maskedF

class VideoSequenceData():

  def __init__(self):
    self.predictedBox = [0,0,0,0]
    self.prevBox = [0,0,0,0]
    self.box = [0,0,0,0]
    self.prv = None
    self.now = None

  def prepareSequence(self):
    self.dataSource = ts.TrajectorySimulator(scene, obj, box, polygon, camera=cam)
    self.deltaW = float(imgSize)/self.dataSource.getFrame().size[0]
    self.deltaH = float(imgSize)/self.dataSource.getFrame().size[1]
    b = self.dataSource.getBox()
    self.box = map(int, [b[0]*self.deltaW, b[1]*self.deltaH, b[2]*self.deltaW, b[3]*self.deltaH])
    self.prevBox = map(int, [b[0]*self.deltaW, b[1]*self.deltaH, b[2]*self.deltaW, b[3]*self.deltaH])
    self.transformFrame()
    self.prev = self.now.copy()
    self.time = 0

  def nextStep(self):
    self.prevBox = map(lambda x:x, self.box)
    step = self.dataSource.nextStep()
    b = self.dataSource.getBox()
    self.box = map(int, [b[0]*self.deltaW, b[1]*self.deltaH, b[2]*self.deltaW, b[3]*self.deltaH])
    self.time += 1
    return step

  def getFrame(self, mode='training', savePath=None):
    if savePath is not None:
      savePath += '/' + str(self.time).zfill(4) + '.jpg'

    self.prv = self.now
    self.transformFrame(save=savePath, box=self.prevBox)
    if cv2 is not None:
      flow = cv2.calcOpticalFlowFarneback(self.prv, self.now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
      flow = None

    if mode == 'training':
      return maskFrame(self.now, flow, self.box)
    else:
      return maskFrame(self.now, flow, self.predictedBox)

  def getMove(self):
    delta = [int(self.box[i]-self.prevBox[i]) for i in range(len(self.box))]
    return delta

  def setPredictedBox(self, delta):
    self.predictedBox = [self.box[i] + delta[i] for i in range(len(self.box))]

  def transformFrame(self, save=None, box=None):
    frame = self.dataSource.getFrame()
    frame = frame.convert('L')
    frame = frame.resize((imgSize,imgSize),Image.ANTIALIAS)
    '''if box is not None:
      draw = ImageDraw.Draw(frame)
      for f in [1,0.75,0.5,0.25]:
        draw.rectangle(fraction(box,f),outline=255)'''
    if save is not None:
      frame.save(save)
    self.now = np.array(frame)

