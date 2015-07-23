import os,sys
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance
import skimage.segmentation

#################################
# GENERATION OF COSINE FUNCTIONS
#################################
MIN_AMPLITUDE = 0.2
MAX_AMPLITUDE = 1.5
MIN_PERIOD = 0.25
MAX_PERIOD = 2.0
MIN_PHASE = 0.0
MAX_PHASE = 1.0
MIN_VSHIFT = -0.5
MAX_VSHIFT = 0.5
RANGE = np.arange(0.0, 6.0, 0.1)

def stretch(values, z1, z2):
  mi = min(values)
  ma = max(values)
  return (z2 - z1)*( (values-mi)/(ma-mi) ) + z1


def cosine(y1, y2):
    a = (MAX_AMPLITUDE - MIN_AMPLITUDE)*np.random.rand() + MIN_AMPLITUDE
    b = (MAX_PERIOD - MIN_PERIOD)*np.random.rand() + MIN_PERIOD
    c = (MAX_PHASE - MIN_PHASE)*np.random.rand() + MIN_PHASE
    d = (MAX_VSHIFT - MIN_VSHIFT)*np.random.rand() + MIN_VSHIFT

    f = a*np.cos(b*RANGE - c) + d
    return stretch(f, y1, y2)

#################################
# TRAJECTORY CLASS
#################################

class Trajectory():

  def __init__(self, w, h):
    # Do sampling of starting and ending points (fixed number of steps).
    # Implicitly selects speed, length and direction of movement.
    # Assume constant speed (no acceleration).
    x1 = (0.8*w - 0.2*w)*np.random.rand() + 0.2*w
    y1 = (0.8*h - 0.2*h)*np.random.rand() + 0.2*h
    x2 = (0.8*w - 0.2*w)*np.random.rand() + 0.2*w
    y2 = (0.8*h - 0.2*h)*np.random.rand() + 0.2*h
    print 'Trajectory: from',int(x1),int(y1),'to',int(x2),int(y2)

    # Sample direction of waving
    if np.random.rand() > 0.5:
      # Horizontal steps, vertical wave
      self.X = stretch(RANGE, x1, x2)
      self.Y = cosine(y1, y2)
    else:
      # Horizontal wave, vertical steps
      self.X = cosine(x1, x2)
      self.Y = stretch(RANGE, y1, y2)

  def getCoord(self, j):
    return (self.X[j], self.Y[j])

#################################
# TRANSFORMATION CLASS
#################################

class Transformation():

  def __init__(self, f, a, b):
    # Initialize range of transformation
    alpha = (b - a)*np.random.rand() + a
    beta = (b - a)*np.random.rand() + a
    if alpha > beta:
      c = alpha
      alpha = beta
      beta = c
    self.func = f
    # Generate a transformation "path"
    self.X = cosine(alpha, beta)

  def transformContent(self, img, j):
    return self.func(img, self.X[j])

  def transformShape(self, w, h, j):
    return self.func(w, h, self.X[j])

#################################
# CONTENT TRANSFORMATIONS
#################################

def identityContent(img, param):
  return img

def rotation(img, angle):
  rot = img.resize( (img.size[0]+10,img.size[1]+10), Image.ANTIALIAS)
  rot.rotate(angle, resample=Image.BICUBIC)
  return rot.resize(img.size, Image.ANTIALIAS)

def color(img, value):
  enhancer = ImageEnhance.Color(img)
  return enhancer.enhance(value)

def contrast(img, value):
  enhancer = ImageEnhance.Contrast(img)
  return enhancer.enhance(value)

def brightness(img, value):
  enhancer = ImageEnhance.Brightness(img)
  return enhancer.enhance(value)

def sharpness(img, value):
  enhancer = ImageEnhance.Sharpness(img)
  return enhancer.enhance(value)

#################################
# SHAPE TRANSFORMATIONS
#################################

MIN_BOX_SIDE = 20

def identityShape(w, h, factor):
  return (w, h)

def scale(w0, h0, factor):
  w = w0 + np.sign(factor)*w0*abs(factor)
  h = h0 + np.sign(factor)*h0*abs(factor)
  return (int(max(w,MIN_BOX_SIDE)),int(max(h,MIN_BOX_SIDE)))

def aspectRatio(w0, h0, factor):
  w,h = w0, h0
  if factor > 1:
    h = h0 + h0*(factor-1)
  else:
    w = w0 + w0*(1-factor)
  return (int(max(w,MIN_BOX_SIDE)),int(max(h,MIN_BOX_SIDE)))

#################################
# OCCLUSSIONS
#################################

class OcclussionGenerator():

  def __init__(self, w, h, maxSize):
    num = np.random.randint(10)
    self.boxes = []
    for i in range(num):
      x1 = (w - maxSize)*np.random.rand()
      y1 = (h - maxSize)*np.random.rand()
      wb = maxSize*np.random.rand()
      hb = maxSize*np.random.rand()
      box = map(int, [x1, y1, x1+wb, y1+hb])
      self.boxes.append(box)

  def occlude(self, img, source):
    for b in self.boxes:
      patch = source.crop(b)
      img.paste(patch, b)
    return img

#################################
# TRAJECTORY SIMULATOR CLASS
#################################

class TrajectorySimulator():

  def __init__(self, sceneFile, objectFile, box, maxSegments=9):
    # Load images
    self.scene = Image.open(sceneFile)
    self.obj = Image.open(objectFile)
    self.obj = self.obj.crop(box)
    objectArray = np.asarray(self.obj)
    nSegments = np.random.randint(2,maxSegments)
    segments = skimage.segmentation.slic(objectArray, n_segments=nSegments)
    #use one segment as mask
    selectedSegment = np.random.randint(nSegments)
    segments[segments != selectedSegment] = 0
    segments[segments == selectedSegment] = 255
    mask = np.uint8(segments)
    #add alpha channel using selected segment mask
    alphaObject = np.dstack((objectArray, mask))
    self.obj = Image.fromarray(alphaObject)
    self.objSize = self.obj.size
    self.box = [0,0,0,0]
    self.step = 0
    # Initialize transformations
    self.shapeTransforms = [
      Transformation(identityShape, 1, 1),
      Transformation(scale, -0.02, 0.02),
      Transformation(aspectRatio, 0.98, 1.02) ]
    self.contentTransforms = [
      Transformation(identityContent, 1, 1),
      Transformation(rotation, -3, 3),
      Transformation(color, 0.60, 1.0),
      Transformation(contrast, 0.60, 1.0),
      Transformation(brightness, 0.60, 1.0),
      Transformation(sharpness, 0.80, 1.2) ]
    self.transform( len(self.contentTransforms) )
    random.shuffle( self.shapeTransforms )
    random.shuffle( self.contentTransforms )
    # Start trajectory
    self.scaleObject()
    self.occluder = OcclussionGenerator(self.scene.size[0], self.scene.size[1], min(self.objSize)*0.5)
    self.trajectory = Trajectory(self.scene.size[0], self.scene.size[1])
    self.render()
    print '@TrajectorySimulator: New simulation with scene {} and object {}:{}'.format(sceneFile, objectFile, box)

  def scaleObject(self):
    # Initial scale of the object is 
    # a fraction of the smallest side of the scene
    smallestSide = min(self.scene.size)
    side = (0.5*smallestSide-0.2*smallestSide)*np.random.rand() + 0.2*smallestSide
    # Preserve object's aspect ratio with the largest side being "side"
    ar = float(self.obj.size[1])/float(self.obj.size[0])
    if self.obj.size[1] > self.obj.size[0]:
      h = side
      w = side/ar
    else:
      h = side*ar
      w = side
    self.objView = self.obj.resize((int(w),int(h)), Image.ANTIALIAS)
    self.objSize = self.objView.size

  def transform(self, top=2):
    self.objSize = self.shapeTransforms[0].transformShape(self.objSize[0], self.objSize[1], self.step)
    self.objView = self.obj.resize(self.objSize, Image.ANTIALIAS)
    for i in range(top):
      newObj = self.contentTransforms[i].transformContent(self.objView, self.step)
      self.objView = newObj

  def render(self):
    self.sceneView = self.scene.copy()
    x = self.trajectory.X[self.step] - 0.5*self.objSize[0]
    y = self.trajectory.Y[self.step] - 0.5*self.objSize[1]
    #paste using alpha channel as mask
    self.sceneView.paste(self.objView, (int(x),int(y)), self.objView.split()[-1])
    self.sceneView = self.occluder.occlude(self.sceneView, self.scene)
    self.box = [max(x,0), max(y,0), min(x+self.objSize[0], self.scene.size[0]), min(y+self.objSize[1],self.scene.size[1])]
    
  def nextStep(self):
    if self.step < self.trajectory.X.shape[0]:
      self.transform()
      self.render()
      self.step += 1
      return True
    else:
      return False

  def saveFrame(self, outDir):
    fname = outDir + '/img/' + str(self.step).zfill(4) + '.jpg'
    self.sceneView.save(fname)
    if self.step <= 1:
      out = open(outDir + '/groundtruth_rect.txt', 'w')
    else:
      out = open(outDir + '/groundtruth_rect.txt', 'a')
    box = map(int,[self.box[0], self.box[1], self.box[2]-self.box[0], self.box[3]-self.box[1]])
    out.write( ' '.join(map(str,box)) + '\n' )
    out.close()

  def convertToGif(self, sequenceDir):
    os.system('convert -delay 1x30 ' + sequenceDir + '/*jpg ' + sequenceDir + '/animation.gif')
    os.system('rm ' + sequenceDir + '*jpg')

## Recommended Usage:
# o = TrajectorySimulator('bogota.jpg','crop_vp.jpg',[0,0,168,210])
# while o.nextStep(): o.saveFrame(dir)
# o.sceneView
