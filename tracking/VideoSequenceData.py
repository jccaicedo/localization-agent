import os
import numpy as np
import PIL.Image as Image
import TrajectorySimulator as ts

#TODO: Put this configuration in an external file or rely entirely on Coco's data
scene = '/home/jccaicedoru/Pictures/test/bogota.jpg'
obj = '/home/jccaicedoru/Pictures/test/photo.jpg'
box = [0, 100, 0, 100]
polygon = [50, 0, 100, 50, 50, 100, 0, 50]
cam = False

MAX_SPEED_PIXELS = 10.0

def transformFrame(source):
  frame = source.getFrame()
  frame = frame.convert('L')
  frame = frame.resize((224,224),Image.ANTIALIAS)
  return np.array(frame)

def maskFrame(frame, box):
  maskedF = np.zeros( (2, frame.shape[0], frame.shape[1]) )
  maskedF[0,:,:] = (frame - 128.0)/128.0
  b = map(int, box)
  maskedF[-1,:,:] = 0
  maskedF[-1,b[0]:b[2],b[1]:b[3]] = 1
  return maskedF

class VideoSequenceData():

  def __init__(self):
    self.predictedBox = [0,0,0,0]
    self.prevBox = [0,0,0,0]
    self.box = [0,0,0,0]

  def prepareSequence(self):
    self.dataSource = ts.TrajectorySimulator(scene, obj, box, polygon, camera=cam)

  def nextStep(self):
    self.prevBox = map(lambda x:x, self.box)
    step = self.dataSource.nextStep()
    self.box = self.dataSource.getBox()
    return step

  def getFrame(self, mode='training'):
    frame = transformFrame(self.dataSource)
    if mode == 'training':
      return maskFrame(frame, self.box)
    else:
      return maskFrame(frame, self.predictedBox)

  def getMove(self):
    delta = [int(self.box[i]-self.prevBox[i])/MAX_SPEED_PIXELS for i in range(len(self.box))]
    return delta

  def setPredictedBox(self, delta):
    self.predictedBox = [self.box[i] + delta[i] for i in range(len(self.box))]

