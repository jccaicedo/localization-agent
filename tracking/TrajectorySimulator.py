import os,sys
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw
#import skimage.segmentation
import numpy.linalg
#import pycocotools.coco

def segmentCrop(image, polygon):
    cropMask = Image.new('L', image.size, 0)
    maskDraw = ImageDraw.Draw(cropMask)
    maskDraw.polygon(polygon, fill=255)
    bounds = polygonBounds(polygon)
    imageCopy = image.copy()
    imageCopy.putalpha(cropMask)
    crop = imageCopy.crop(bounds)
    return crop

def polygonBounds(polygon):
    maskCoords = numpy.array(polygon).reshape(len(polygon)/2,2).T
    bounds = map(int, (maskCoords[0].min(), maskCoords[1].min(), maskCoords[0].max(), maskCoords[1].max()))
    return bounds

###############################
#Sampler for affine transformations
###############################

class AffineSampler(object):

    def __init__(self, scale=[1,1], angle=0, translation=[0,0], *args, **kwargs):
        self.scale = numpy.array(scale)
        self.angle = numpy.array(angle)
        self.translation = numpy.array(translation)
        super(AffineSampler, self).__init__(*args, **kwargs)

    def sample(self):
        self.scale = 0.1*numpy.random.standard_normal(self.scale.shape)+self.scale
        self.angle = numpy.pi/36*numpy.random.standard_normal(self.angle.shape)+self.angle
        self.translation = numpy.random.standard_normal(self.translation.shape)+self.translation

    def __repr__(self):
        return 'Scale: {}\tAngle: {}\tTranslation: {}'.format(self.scale, self.angle, self.translation)

    def transform(self):
        #The order is scale, rotate and translate
        return numpy.dot(self.applyTranslate(self.translation), numpy.dot(self.applyRotate(self.angle), self.applyScale(self.scale)))

    def applyScale(self, scales):
        return numpy.array([[scales[0], 0, 0],[0, scales[1], 0],[0, 0, 1]])

    def applyRotate(self, angle):
        return numpy.array([[numpy.cos(angle), numpy.sin(angle), 0],[-numpy.sin(angle), numpy.cos(angle), 0],[0, 0, 1]])

    def applyTranslate(self, translation):
        return numpy.array([[1, 0, translation[0]],[0,1,translation[1]],[0, 0, 1]])

    def applyTransform(self, crop):
        size = crop.size
        cropCorners = numpy.array([[0, 0, 1],[size[0], 0, 1],[size[0], size[1], 1],[0, size[1], 1]]).T
        transformedCorners = numpy.dot(self.transform(), cropCorners)
        left, upper, right, lower = map(int,(transformedCorners[0,:].min(), transformedCorners[1,:].min(), transformedCorners[0,:].max(), transformedCorners[1,:].max()))
        newSize = (right-left, lower-upper)
        correctedTransform = numpy.dot(self.applyTranslate([-left, -upper]), self.transform())
        return crop.transform(newSize, Image.AFFINE, tuple(numpy.linalg.inv(correctedTransform).flatten()[:6]))

    def pasteCrop(self, image, box, crop):
        imageCopy = image.copy()
        imageCopy.paste(crop, box=box, mask=crop)
        return imageCopy

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
  rot = rot.rotate(angle, resample=Image.BICUBIC)
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

  def __init__(self, sceneFile, objectFile, box, polygon=None, maxSegments=9, camSize=None, axes=False):
    # Load images
    self.scene = Image.open(sceneFile)
    self.obj = Image.open(objectFile)
    if camSize is None:
        camSize = self.scene.size
    self.camSize = camSize
    if polygon is None:
        polygon = (box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3])
    self.obj = segmentCrop(self.obj, polygon)
    if axes:
      self.scene = self.draw_axes(self.scene)
      self.obj = self.draw_axes(self.obj)
    self.objSize = self.obj.size
    self.prevBox = [0,0,0,0]
    self.box = [0,0,0,0]
    self.step = 0
    # Initialize transformations
    #TODO: select adequate values for transforms and maybe sample them from a given distribution
    self.shapeTransforms = [
      Transformation(identityShape, 1, 1),
      Transformation(scale, -0.02, 0.02),
      Transformation(aspectRatio, 0.98, 1.02) ]
    self.contentTransforms = [
      Transformation(identityContent, 1, 1),
      Transformation(rotation, -30, 30),
      #TODO: reenable but check if they are the culprit for transparency cases
      #Transformation(color, 0.60, 1.0),
      #Transformation(contrast, 0.60, 1.0),
      #Transformation(brightness, 0.60, 1.0),
      #Transformation(sharpness, 0.80, 1.2)
    ]
    #TODO: include cropping
    self.cameraContentTransforms = [
      Transformation(rotation, -10, 10),
      Transformation(identityContent, 1, 1),
    ]
    self.cameraShapeTransforms = [
      Transformation(identityShape, 1, 1),
      Transformation(scale, -0.1, 0.1),
    ]
    self.transform( len(self.contentTransforms) )
    random.shuffle( self.shapeTransforms )
    random.shuffle( self.contentTransforms )
    # Start trajectory
    self.scaleObject()
    self.occluder = OcclussionGenerator(self.scene.size[0], self.scene.size[1], min(self.objSize)*0.5)
    self.trajectory = Trajectory(self.scene.size[0], self.scene.size[1])
    self.render()
    print '@TrajectorySimulator: New simulation with scene {} and object {}'.format(sceneFile, objectFile)

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
    self.sceneView.paste(self.objView, (int(x),int(y)), self.objView)
    self.sceneView = self.occluder.occlude(self.sceneView, self.scene)
    self.prevBox = map(lambda x:x, self.box)
    for i in range(len(self.cameraShapeTransforms)):
      self.sceneSize = self.cameraShapeTransforms[i].transformShape(self.scene.size[0], self.scene.size[1], self.step)
      self.sceneView = self.sceneView.resize(self.sceneSize, Image.ANTIALIAS).crop((0,0) + self.scene.size)
    for i in range(len(self.cameraContentTransforms)):
      newScene = self.cameraContentTransforms[i].transformContent(self.sceneView, self.step)
      self.sceneView = newScene
    #TODO: adjust box coordinates according to camera transforms
    cameraCorners = map(int, (self.trajectory.X[self.step]-0.5*self.camSize[0],self.trajectory.Y[self.step]-0.5*self.camSize[1],self.trajectory.X[self.step]+0.5*self.camSize[0],self.trajectory.Y[self.step]+0.5*self.camSize[1]))
    self.camView = self.sceneView.crop(cameraCorners)
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
    fname = os.path.join(outDir, str(self.step).zfill(4) + '.jpg')
    self.sceneView.save(fname)
    gtPath = os.path.join(outDir, 'groundtruth_rect.txt')
    if self.step <= 1:
      out = open(gtPath, 'w')
    else:
      out = open(gtPath, 'a')
    box = map(int,[self.box[0], self.box[1], self.box[2]-self.box[0], self.box[3]-self.box[1]])
    out.write( ' '.join(map(str,box)) + '\n' )
    out.close()

  def getMaskedFrame(self, box=None):
    frame = np.asarray(self.sceneView)
    maskedF = np.zeros( (frame.shape[0],frame.shape[1],frame.shape[2]+1) )
    maskedF[:,:,0:frame.shape[2]] = (frame - 128.0)/128.0
    if box is None:
      b = map(int, self.box)
    else:
      b = map(int, box)
    #maskedF[:,:,-1] = -1
    maskedF[b[0]:b[2],b[1]:b[3],-1] = 1
    return maskedF

  def getMove(self):
    delta = [int(self.box[i]-self.prevBox[i]) for i in range(len(self.box))]
    

  def convertToGif(self, sequenceDir):
    os.system('convert -delay 1x30 ' + sequenceDir + '/*jpg ' + sequenceDir + '/animation.gif')
    os.system('rm ' + sequenceDir + '*jpg')

  def __iter__(self):
    return self

  def next(self):
    if self.nextStep():
      return self.camView
    else:
      raise StopIteration()

  def draw_axes(self, image):
    size = image.size
    imageCopy = image.copy()
    draw = ImageDraw.Draw(imageCopy)
    minSize = min(size[1], size[0])
    width = int(minSize*0.1)
    length = int(minSize*0.3)
    draw.line(map(int, (width/2, width/2, width/2, length)), fill=(255, 0, 0), width=width)
    draw.line(map(int, (width/2, width/2, length, width/2)), fill=(0, 255, 0), width=width)
    
    del draw
    return imageCopy

## Recommended Usage:
# o = TrajectorySimulator('bogota.jpg','crop_vp.jpg',[0,0,168,210])
# while o.nextStep(): o.saveFrame(dir)
# o.sceneView

class COCOSimulatorFactory():

    #Assumes standard data layout as specified in https://github.com/pdollar/coco/blob/master/README.txt
    def __init__(self, dataDir, dataType):
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '%s/annotations/instances_%s.json'%(dataDir,dataType)
        self.imagePathTemplate = '%s/images/%s/%s'
        #COCO dataset handler object
        print '!!!!!!!!!!!!! WARNING: Loading the COCO annotations can take up to 3 GB RAM !!!!!!!!!!!!!'
        self.coco = pycocotools.coco.COCO(self.annFile)
        #TODO: Filter the categories to use in sequence generation
        self.catIds = self.coco.getCatIds()
        cats = self.coco.loadCats(self.catIds)
        nms=[cat['name'] for cat in cats]
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.fullImgIds = self.coco.getImgIds()
        print 'Number of categories {} and corresponding images {}'.format(len(self.catIds), len(self.imgIds))
        print 'Category names: {}'.format(', '.join(nms))
        
    def createInstance(self):
        #Select a random image for the scene
        sceneData = self.coco.loadImgs(self.fullImgIds[np.random.randint(0, len(self.fullImgIds))])[0]
        scenePath = self.imagePathTemplate%(self.dataDir, self.dataType, sceneData['file_name'])

        #Select a random image for the object, restricted to annotation categories
        objData = self.coco.loadImgs(self.imgIds[np.random.randint(0, len(self.imgIds))])[0]
        objPath = self.imagePathTemplate%(self.dataDir, self.dataType, objData['file_name'])

        #Get annotations for object scene
        objAnnIds = self.coco.getAnnIds(imgIds=objData['id'], catIds=self.catIds, iscrowd=None)
        objAnns = self.coco.loadAnns(objAnnIds)

        #Select a random object in the scene and read the segmentation polygon
        objectAnnotations = objAnns[np.random.randint(0, len(objAnns))]
        print 'Segmenting object from category {}'.format(self.coco.loadCats(objectAnnotations['category_id'])[0]['name'])
        polygon = objectAnnotations['segmentation'][np.random.randint(0, len(objectAnnotations['segmentation']))]

        scene = Image.open(scenePath)
        camSize = map(int, (scene.size[0]*0.5, scene.size[1]*0.5)) 
        scene.close()
        #Does not work as expected due to object scaling on range 0.2-0.5 of smallest scene side
        #objBounds = polygonBounds(polygon)
        #camFactor = 2+2*np.random.rand()
        #camSize = map(int, ((objBounds[2]-objBounds[0])*camFactor, (objBounds[3]-objBounds[1])*camFactor))
        #print 'Object bounds are {} and camera factor is {}, resulting camera size is {}'.format(objBounds, camFactor, camSize)
        simulator = TrajectorySimulator(scenePath, objPath, [], polygon=polygon, camSize=camSize)
        
        return simulator

    def create(self, sceneFullPath, objectFullPath, axes=False):
        #TODO: make really definite
        sceneDict = [data for data in self.coco.loadImgs(self.fullImgIds) if str(data['file_name']) == os.path.basename(sceneFullPath)][0]
        objectDict = [data for data in self.coco.loadImgs(self.imgIds) if str(data['file_name']) == os.path.basename(objectFullPath)][0]
        scenePath = self.imagePathTemplate%(self.dataDir, self.dataType, sceneDict['file_name'])
        objPath = self.imagePathTemplate%(self.dataDir, self.dataType, objectDict['file_name'])
        objAnnIds = self.coco.getAnnIds(imgIds=objectDict['id'], catIds=self.catIds, iscrowd=None)
        objAnns = self.coco.loadAnns(objAnnIds)
        objectAnnotations = objAnns[np.random.randint(0, len(objAnns))]
        print 'Segmenting object from category {}'.format(self.coco.loadCats(objectAnnotations['category_id'])[0]['name'])
        polygon = objectAnnotations['segmentation'][np.random.randint(0, len(objectAnnotations['segmentation']))]
        scene = Image.open(scenePath)
        camSize = map(int, (scene.size[0]*0.5, scene.size[1]*0.5))
        scene.close()

        simulator = TrajectorySimulator(scenePath, objPath, [], polygon=polygon, camSize=camSize, axes=axes)

        return simulator
