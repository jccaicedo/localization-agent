import os
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import TrajectorySimulator as ts
import TraxClient as tc
import TrackerState as bs

#TODO: Put this configuration in an external file or rely entirely on Coco's data
box = [0, 100, 0, 100]
polygon = [50, 0, 100, 50, 50, 100, 0, 50]

imgSize = 224
channels = 3
totalFrames = 60
cam = False

def fraction(b,k):
  w = (b[2]-b[0])*(1-k)/2.
  h = (b[3]-b[1])*(1-k)/2.
  return [b[0]+w,b[1]+h, b[2]-w,b[3]-h]

def normalize(img, box=None):
  if box is not None:
    view = img.crop(map(int,box))
    view.load()
  else:
    view = img.copy()
  view = view.convert('RGB')
  view = view.resize((imgSize,imgSize),Image.ANTIALIAS)
  view = (np.array(view) - 128)/128
  view = np.swapaxes(np.swapaxes(view, 0, 2), 1, 2)
  return view

class SearchSequenceGenerator(object):

  def __init__(self, image, save=None):
    self.image = image
    self.views = []
    self.actions = []
    self.time = 0
    self.save = save

  def generateSequence(self, pairView, targetBox):
    self.views.append( pairView )            # Add the target object to the sequence
    self.actions.append( bs.TARGET )         # Mark object as target
    searcher = bs.TrackerState(self.image,target=targetBox)
    a = searcher.sampleBestAction()          # Given the full scene, take the best action
    self.moveOneStep(searcher.box, pairView) # Record the first view
    self.actions.append( a )                 # Record the movement given the view
    while a != bs.PLACE_LANDMARK and a != bs.ABORT:
      box = searcher.performAction(a)  # Move the box according to the previous decision
      a = searcher.sampleBestAction()  # Given the current view, take the best action
      self.moveOneStep(box, pairView)  # Record the current view
      self.actions.append( a )         # Record the next movement given the current view
    if a == bs.PLACE_LANDMARK:
      self.moveOneStep(box, pairView)  # Duplicate the last view
      self.actions.append( a )         # End the sequence with the terminal action
    else:
      self.views, self.actions = [],[]
    return self.views, self.actions

  def moveOneStep(self, box, pair):
    # Duplicated reference view
    #data = np.zeros( (2,pair.shape[0],pair.shape[1],pair.shape[2]) )
    #data[0,:,:,:] = pair
    #data[1,:,:,:] = normalize(self.image, box)
    # No duplicated reference view
    data = np.zeros( (pair.shape[0],pair.shape[1],pair.shape[2]) )
    data[:,:,:] = normalize(self.image, box)
    self.views.append( data )
    self.time += 1

class BoxSearchSequenceData(object):

  def __init__(self, workingDir='/home/jccaicedo/data/tracking/simulations/debug/', sequenceList=None):
    self.predictedBox = [0,0,0,0]
    self.prevBox = [0,0,0,0]
    self.box = [0,0,0,0]
    self.scene = None
    self.targetView = None
    self.workingDir = workingDir
    self.sequenceList = sequenceList

  def prepareSequence(self, loadSequence=None):
    if loadSequence is None:
      scene = self.workingDir + 'bogota.jpg'
      obj = self.workingDir + 'photo.jpg'
      #TODO: convert box to polygon
      self.dataSource = ts.TrajectorySimulator(scene, obj, polygon, camera=cam)
    elif loadSequence == 'list' and self.sequenceList is not None:
      self.dataSource = self.sequenceList.pop()
      self.dataSource.start()
    elif loadSequence == 'TraxClient':
      self.dataSource = TraxClientWrapper('/home/fhdiaze/TrackingAgent/Benchmarks/vot-toolkit/tracker/examples/native/libvot.so')
    else:
      self.dataSource = StaticDataSource(loadSequence) 
    self.deltaW = float(imgSize)/self.dataSource.getFrame().size[0]
    self.deltaH = float(imgSize)/self.dataSource.getFrame().size[1]
    b = self.dataSource.getBox()
    self.box = map(int, [b[0]*self.deltaW, b[1]*self.deltaH, b[2]*self.deltaW, b[3]*self.deltaH])
    self.prevBox = self.box[:]
    self.predictedBox = self.box[:]
    self.transformFrame()
    self.time = 0

  def nextStep(self, mode='training'):
    self.prevBox = self.box[:]
    end = self.dataSource.nextStep()
    if mode == 'training':
      b = self.dataSource.getBox()
      self.box = map(int, [b[0]*self.deltaW, b[1]*self.deltaH, b[2]*self.deltaW, b[3]*self.deltaH])
    else:
      b = self.dataSource.getBox()
      tmp = map(int, [b[0]*self.deltaW, b[1]*self.deltaH, b[2]*self.deltaW, b[3]*self.deltaH])
      print 'Predicted:',self.predictedBox, 'Real:',tmp,'Diff:',[tmp[i]-self.predictedBox[i] for i in range(len(tmp))]
      self.box = self.predictedBox[:]
    self.time += 1
    return end

  def getFrame(self, savePath=None):
    if savePath is not None:
      with open(savePath + '/rects.txt','a') as rects:
        rects.write(' '.join(map(str,self.box)) + '\n')
      savePath += '/' + str(self.time).zfill(4) + '.jpg'

    self.transformFrame(save=savePath, box=self.prevBox)
    return self.now['views']

  def getMove(self):
    return self.now['actions']

  def setMove(self, delta):
    print 'Delta:',delta
    self.dataSource.reportBox(self.predictedBox)

  def transformFrame(self, save=None, box=None):
    frame = self.dataSource.getFrame()
    frame = frame.resize((imgSize,imgSize),Image.ANTIALIAS)
    '''if box is not None:
      draw = ImageDraw.Draw(frame)
      for f in [1,0.75,0.5,0.25]:
        draw.rectangle(fraction(box,f),outline=255)
    if save != None:
      frame.save(save)'''
    # Prepare the target in the first frame
    if self.targetView is None:
      self.targetView = normalize(frame, box)
    # Compute the sequence of box transformations to reach the target
    ssg = SearchSequenceGenerator(frame,save=save)
    views, actions = ssg.generateSequence(self.targetView, self.box)
    self.now = {'views':views, 'actions':actions}
    # The target for the next frame is initialized here
    self.targetView = normalize(frame, box)


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

  def __init__(self, libvotPath):
    self.client = tc.TraxClient(libvotPath)
    self.path = self.client.nextFramePath()
    self.box = self.client.initialize()
    # TODO: Transform box from 4 coordinates to 2 coordinates

  def getFrame(self):
    img = Image.open(self.path)
    return img

  def getBox(self):
    return self.box

  def reportBox(self, box):
    # TODO: Transform from 2 coordinates to 4 coordinates
    self.box = box
    self.client.reportRegion(box)

  def nextStep(self):
    self.path = self.client.nextFramePath()
    if self.path == '':
      self.client.quit()
      return False
    else:
      return True
