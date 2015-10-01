import os,sys
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw
import numpy.linalg

def segmentCrop(image, polygon):
    cropMask = Image.new('L', image.size, 0)
    maskDraw = ImageDraw.Draw(cropMask)
    maskDraw.polygon(polygon, fill=255)
    bounds = polygon_bounds(polygon)
    imageCopy = image.copy()
    imageCopy.putalpha(cropMask)
    crop = imageCopy.crop(bounds)
    return crop

def polygon_bounds(polygon):
    maskCoords = np.array(polygon).reshape(len(polygon)/2,2).T
    bounds = map(int, (maskCoords[0].min(), maskCoords[1].min(), maskCoords[0].max(), maskCoords[1].max()))
    return bounds

def applyScale(scales):
    return np.array([[scales[0], 0, 0],[0, scales[1], 0],[0, 0, 1]])

def applyRotate(angle):
    return np.array([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0],[0, 0, 1]])

def applyTranslate(translation):
    return np.array([[1, 0, translation[0]],[0,1,translation[1]],[0, 0, 1]])

def applyTransform(crop, transform, camSize):
    # Requires inverse as the parameters transform from object to camera 
    return crop.transform(camSize, Image.AFFINE, np.linalg.inv(transform).flatten()[:7])

# Points must be in homogeneous coordinates
def transform_points(transform, points):
    transformedCorners = np.dot(transform, points)
    return transformedCorners
    
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

  def __init__(self, f, a, b, pathFunction=None, steps=64):
    self.func = f
    if pathFunction is None:
        # Initialize range of transformation
        alpha = (b - a)*np.random.rand() + a
        beta = (b - a)*np.random.rand() + a
        if alpha > beta:
          c = alpha
          alpha = beta
          beta = c
        # Generate a transformation "path"
        self.X = cosine(alpha, beta)
    else:
        self.X = pathFunction(a, b, steps)

  def transformContent(self, img, j):
    return self.func(img, self.X[j])

  def transformShape(self, w, h, j):
    return self.func(w, h, self.X[j])

#################################
# CONTENT TRANSFORMATIONS
#################################

def rotation(img, angle):
  matrix = applyRotate(angle)
  return matrix

def translateX(img, value):
  matrix = applyTranslate([value, 0])
  return matrix

def translateY(img, value):
  matrix = applyTranslate([0, value])
  return matrix

def scaleX(img, value):
  matrix = applyScale([value, 1])
  return matrix

def scaleY(img, value):
  matrix = applyScale([1, value])
  return matrix

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

  def __init__(self, sceneFile, objectFile, box, polygon=None, maxSegments=9, camSize=None, axes=False, maxSteps=None, contentTransforms=None, shapeTransforms=None, cameraContentTransforms=None, cameraShapeTransforms=None, drawBox=False, camera=True, drawCam=False):
    if maxSteps is None:
        maxSteps = len(RANGE)
    self.maxSteps = maxSteps
    # Load images
    self.scene = Image.open(sceneFile)
    self.obj = Image.open(objectFile)
    # Use scene as camera
    if camSize is None:
        camSize = self.scene.size
    self.camSize = camSize
    # Correct camera size to be even as needed by video encoding software
    evenCamSize = list(self.camSize)
    for index in range(len(evenCamSize)):
        if evenCamSize[index] % 2 ==1:
            evenCamSize[index] += 1
    self.camSize = tuple(evenCamSize) 
    # Use box as polygon
    if polygon is None:
        polygon = (box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3])
    self.polygon = polygon
    self.drawBox = drawBox
    self.drawCam = drawCam
    self.camera = camera
    #Segment the object using the polygon and crop to the resulting axes-aligned bounding box
    self.obj = segmentCrop(self.obj, polygon)
    # Draw coordinate axes for each source
    if axes:
      self.scene = self.draw_axes(self.scene)
      self.obj = self.draw_axes(self.obj)
    self.objSize = self.obj.size
    self.prevBox = [0,0,0,0]
    self.box = [0,0,0,0]
    self.step = 0
    self.validStep = 0
    # Initialize transformations
    #TODO: select adequate values for transforms and maybe sample them from a given distribution
    if shapeTransforms is None:
        self.shapeTransforms = [
            Transformation(identityShape, 1, 1),
        ]
    else:
        self.shapeTransforms = shapeTransforms
    if contentTransforms is None:
        self.contentTransforms = [
            Transformation(scaleX, 0.5, 2),
            Transformation(scaleY, 0.5, 2),
            Transformation(rotation, -np.pi, np.pi),
            Transformation(translateX, 0, self.scene.size[0]),
            Transformation(translateY, 0, self.scene.size[1]),
            #TODO: reenable but check if they are the culprit for transparency cases
            #Transformation(color, 0.60, 1.0),
            #Transformation(contrast, 0.60, 1.0),
            #Transformation(brightness, 0.60, 1.0),
            #Transformation(sharpness, 0.80, 1.2)
        ]
    else:
        self.contentTransforms = contentTransforms
    if cameraContentTransforms is None:   
        self.cameraContentTransforms = [
        ]
    else:
        self.cameraContentTransforms = cameraContentTransforms
    if cameraShapeTransforms is None:
        self.cameraShapeTransforms = [
            Transformation(identityShape, 1, 1),
        ]
    else:
        self.cameraShapeTransforms = cameraShapeTransforms
    # Start trajectory
    self.scaleObject()
    # Calculate bounds after scaling
    self.bounds = np.array([[0,self.objSize[0],self.objSize[0],0],[0,0,self.objSize[1],self.objSize[1]]])
    self.bounds = np.vstack([self.bounds, np.ones((1,self.bounds.shape[1]))])
    self.cameraBounds = np.array([[0,self.camSize[0],self.camSize[0],0],[0,0,self.camSize[1],self.camSize[1]]])
    self.cameraBounds = np.vstack([self.cameraBounds, np.ones((1,self.cameraBounds.shape[1]))])
    self.occluder = OcclussionGenerator(self.scene.size[0], self.scene.size[1], min(self.objSize)*0.5)
    self.currentTransform = np.eye(3,3)
    self.cameraTransform = np.eye(3,3)
    self.transform( len(self.contentTransforms) )
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

  def validate_bounds(self, transform, points, size):
    transformedPoints = transform_points(transform, points)
    return np.all(np.logical_and(np.greater(transformedPoints[:2,:], [[0], [0]]), np.less(transformedPoints[:2,:], [[size[0]],[size[1]]])))

  def transform(self, top=2):
    self.objSize = self.shapeTransforms[0].transformShape(self.objSize[0], self.objSize[1], self.step)
    self.objView = self.obj.resize(self.objSize, Image.ANTIALIAS)
    # Concatenate transforms and apply them to obtain transformed object
    newMatrix = np.eye(3,3)
    for i in range(top):
      newMatrix = np.dot(self.contentTransforms[i].transformContent(self.objView, self.step), newMatrix)
    # Only update if valid
    if self.validate_bounds(newMatrix, self.bounds, self.scene.size):
        self.currentTransform = newMatrix
        self.validStep = self.step
    self.objView = applyTransform(self.objView, self.currentTransform, self.scene.size)

  def render(self):
    self.sceneView = self.scene.copy()
    # Paste the transformed object, at origin as scene is absolute reference system
    self.sceneView.paste(self.objView, (int(0),int(0)), self.objView)
    self.sceneView = self.occluder.occlude(self.sceneView, self.scene)
    self.prevBox = map(lambda x:x, self.box)
    for i in range(len(self.cameraShapeTransforms)):
      self.sceneSize = self.cameraShapeTransforms[i].transformShape(self.scene.size[0], self.scene.size[1], self.step)
      self.sceneView = self.sceneView.resize(self.sceneSize, Image.ANTIALIAS).crop((0,0) + self.scene.size)
    # Concatenate camera transforms
    if self.camera:
        newMatrix = np.eye(3,3)
        for i in range(len(self.cameraContentTransforms)):
            newMatrix = np.dot(self.cameraContentTransforms[i].transformContent(self.sceneView, self.step), newMatrix)
        self.cameraTransform = newMatrix
        # Obtain definite camera transform by appending object transform
        self.camView = applyTransform(self.sceneView, np.dot(self.cameraTransform, np.linalg.inv(self.currentTransform)), self.camSize)
        referenceTransform = self.cameraTransform
    else:
        self.camView = self.sceneView
        referenceTransform = self.currentTransform
    # Obtain bounding box points on camera coordinate system
    boxPoints = transform_points(referenceTransform, self.bounds)
    self.box = [max(min(boxPoints[0,:]),0), max(min(boxPoints[1,:]),0), min(max(boxPoints[0,:]), self.camSize[0]-1), min(max(boxPoints[1,:]),self.camSize[1]-1)]
    if self.drawBox:
        self.camDraw = ImageDraw.ImageDraw(self.camView)
        self.camDraw.rectangle(self.box)
    if self.drawCam:
        camPoints = transform_points(np.dot(self.currentTransform, self.cameraTransform), self.cameraBounds)
        cameraBox = [max(min(camPoints[0,:]),0), max(min(camPoints[1,:]),0), min(max(camPoints[0,:]), self.scene.size[0]-1), min(max(camPoints[1,:]),self.scene.size[1]-1)]
        self.sceneDraw = ImageDraw.ImageDraw(self.sceneView)
        self.sceneDraw.rectangle(cameraBox)
    
  def nextStep(self):
    if self.step < self.maxSteps:
      self.transform( len(self.contentTransforms) )
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

try:
    import pycocotools.coco

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
            
        def createInstance(self, *args, **kwargs):
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
            simulator = TrajectorySimulator(scenePath, objPath, [], polygon=polygon, camSize=camSize, *args, **kwargs)
            
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
except Exception as e:
    print 'No support for pycoco'
