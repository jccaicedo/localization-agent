import os
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import TrajectorySimulator as ts
import TraxClient as tc
import logging
try:
    import cv2
    channels = 4
except:
    cv2 = None
    channels = 2

#TODO: Put this configuration in an external file or rely entirely on Coco's data
box = [0, 100, 0, 100]
polygon = [50, 0, 100, 50, 50, 100, 0, 50]
imgSize = 64
totalFrames = 60
cam = False
maskMove = 6

MAX_SPEED_PIXELS = 1.0

def fraction(b,k):
    w = (b[2]-b[0])*(1-k)/2.
    h = (b[3]-b[1])*(1-k)/2.
    return [b[0]+w,b[1]+h, b[2]-w,b[3]-h]

def makeMask(w,h,box):
    mask = np.zeros((w,h))
    for factor in [(1.00, 1), (0.75, -1), (0.5, 1), (0.25, -1)]:
        b = map(int, fraction(box, factor[0]))
        mask[b[1]:b[3], b[0]:b[2]] = factor[1] # y dimensions first (rows)
    return mask

def maskFrame(frame, flow, box):
    if flow is not None:
        maskedF = np.zeros( (4, frame.shape[0], frame.shape[1]) )
        maskedF[1,:,:] = flow[...,0]
        maskedF[2,:,:] = flow[...,1]
    else:
        maskedF = np.zeros( (2, frame.shape[0], frame.shape[1]) )
    maskedF[0,:,:] = (frame - 128.0)/128.0
    maskedF[-1,:,:] = makeMask(frame.shape[0], frame.shape[1], box)

    return maskedF

def boxToPolygon(box):
    leftTopX = box[0]
    leftTopY = box[1]
    width = box[2] - box[0]
    height = box[3] - box[1]
    rightTopX = leftTopX + width
    rightTopY = leftTopY
    rightBottomX = rightTopX
    rightBottomY = rightTopY + height
    leftBottomX = leftTopX
    leftBottomY = leftTopY + height
            
    coords = [leftTopX, leftTopY, rightTopX, rightTopY, rightBottomX, rightBottomY, leftBottomX, leftBottomY]
    return coords

class VideoSequenceData(object):

    def __init__(self, workingDir='/home/jccaicedo/data/tracking/simulations/debug/'):
        self.predictedBox = [0,0,0,0]
        self.prevBox = [0,0,0,0]
        self.box = [0,0,0,0]
        self.prv = None
        self.now = None
        self.workingDir = workingDir

    def prepareSequence(self, loadSequence=None):
        if loadSequence is None:
          scene = self.workingDir + 'bogota.jpg'
          obj = self.workingDir + 'photo.jpg'
          #TODO: convert box to polygon
          self.dataSource = ts.TrajectorySimulator(scene, obj, polygon, camera=cam)
        elif loadSequence == 'TraxClient':
          self.dataSource = TraxClientWrapper('/home/fhdiaze/TrackingAgent/Benchmarks/vot-toolkit/tracker/examples/native/libvot.so')
        else:
          self.dataSource = StaticDataSource(loadSequence) 
        self.scaleW = float(imgSize)/self.dataSource.getFrame().size[0]
        self.scaleH = float(imgSize)/self.dataSource.getFrame().size[1]
        b = self.dataSource.getBox()
        self.box = map(int, [b[0]*self.scaleW, b[1]*self.scaleH, b[2]*self.scaleW, b[3]*self.scaleH])
        self.prevBox = self.box[:]
        self.predictedBox = self.box[:]
        self.transformFrame()
        self.prev = self.now.copy()
        self.time = 0

    def nextStep(self, mode='training'):
        if self.time % maskMove == 0:
          self.prevBox = self.box[:]
        end = self.dataSource.nextStep()
        if mode == 'training':
          b = self.dataSource.getBox()
          self.box = map(int, [b[0]*self.scaleW, b[1]*self.scaleH, b[2]*self.scaleW, b[3]*self.scaleH])
        else:
          b = self.dataSource.getBox()
          tmp = map(int, [b[0]*self.scaleW, b[1]*self.scaleH, b[2]*self.scaleW, b[3]*self.scaleH])
          #print 'Predicted:',self.predictedBox, 'Real:',tmp,'Diff:',[tmp[i]-self.predictedBox[i] for i in range(len(tmp))]
          self.box = self.predictedBox[:]
        self.time += 1
        return end

    def getFrame(self, savePath=None):
        if savePath is not None:
          with open(savePath + '/rects.txt','a') as rects:
            rects.write(' '.join(map(str,self.box)) + '\n')
          savePath += '/' + str(self.time).zfill(4) + '.jpg'
    
        self.prv = self.now
        self.transformFrame(save=savePath, box=self.prevBox)
        if cv2 is not None:
          flow = cv2.calcOpticalFlowFarneback(self.prv, self.now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
          flow = None
    
        return maskFrame(self.now, flow, self.prevBox)

    def getMove(self):
        delta = [int(self.box[i]-self.prevBox[i])/MAX_SPEED_PIXELS for i in range(len(self.box))]
        return delta

    def setMove(self, delta):
        #print 'Delta:',delta
        b = [self.box[i] + round(delta[i]*MAX_SPEED_PIXELS) for i in range(len(self.box))]
        b = [max(min(x,imgSize-1),0) for x in b]
        rescaledBox = map(int, [b[0]/self.scaleW, b[1]/self.scaleH, b[2]/self.scaleW, b[3]/self.scaleH])
        self.dataSource.reportBox(rescaledBox)
        self.predictedBox = b

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

class StaticDataSource(object):

  def __init__(self, directory):
    data = os.listdir(directory)
    self.dir = directory
    self.frames = [d for d in data if d.endswith(".jpg")]
    self.frames.sort()
    self.boxes = [ map(int,b.split()) for b in open(directory + '/rects.txt')]
    self.img = Image.open(self.dir + self.frames[0])
    self.current = 0
  
  def getFrame(self):
    return self.img

  def getBox(self):
    return self.boxes[self.current]

  def reportBox(self, box):
    return

  def nextStep(self):
    if self.current < len(self.frames)-1:
      self.current += 1
      self.img = Image.open(self.dir + self.frames[self.current])
      return True
    else:
      return False

class TraxClientWrapper(object):
  region = None
  
  def __init__(self, libvotPath):
    self.client = tc.TraxClient(libvotPath)
    self.region = self.client.getInitialRegion()
    logging.info('Initial region: %s', self.getBox())
    self.path = self.client.nextFrame()
    logging.info('Initial path: %s', self.path)

  def getFrame(self):
    img = Image.open(self.path)
    return img

  def getBox(self):
    box = self.region.toList()
        
    sampledBox = [box[i] for i in [0, 1, 4, 5]]
    #Guarantee box ordering
    sampledBox = [np.min(sampledBox[0::2]), np.min(sampledBox[1::2]), np.max(sampledBox[0::2]), np.max(sampledBox[1::2])]

    return sampledBox

  def reportBox(self, box):
    self.box = box
    coords = boxToPolygon(self.box)
    logging.info('Reporting box: %s', coords)
    boxT = tc.Box()
    boxT.setRegion(coords)
        
    self.client.reportRegion(boxT)

  def nextStep(self):
    self.path = self.client.nextFrame()
    logging.info('New path: %s', self.path)
    if self.path == '':
      logging.info('Quitting as new path is empty')
      self.client.quit()
      return False
    else:
      return True
