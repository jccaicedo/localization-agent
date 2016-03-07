import os,sys
import numpy as np
import random, time
from PIL import Image
from PIL import ImageFile
#TODO: Leave False so that an Exception is generated
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageEnhance
from PIL import ImageDraw
import numpy.linalg
import pickle
import logging

def startRandGen():
  r = random.Random()
  r.jumpahead(long(time.time()))
  return r

def segmentCrop(image, polygon):
    '''Segments an image using the given polygon and returns a crop of the bounding box including alpha channel'''
    cropMask = Image.new('L', image.size, 0)
    maskDraw = ImageDraw.Draw(cropMask)
    maskDraw.polygon(polygon, fill=255)
    bounds = polygon_bounds(polygon)
    imageCopy = image.copy()
    imageCopy.putalpha(cropMask)
    crop = imageCopy.crop(bounds)
    return crop

def polygon_bounds(polygon):
    '''Calculates the bounding box for the given polygon'''
    maskCoords = np.array(polygon).reshape(len(polygon)/2,2).T
    bounds = map(int, (maskCoords[0].min(), maskCoords[1].min(), maskCoords[0].max(), maskCoords[1].max()))
    return bounds

def applyScale(scales):
    '''Calculates a scaling transformation matrix with independent scaling'''
    return np.array([[scales[0], 0, 0],[0, scales[1], 0],[0, 0, 1]])

def applyRotate(angle):
    ''''Calculates a rotation transformation matrix from an angle in radians'''
    return np.array([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0],[0, 0, 1]])

def applyTranslate(translation):
    '''Calculates a trans;lation transformation matrix from x and y coordinates'''
    return np.array([[1, 0, translation[0]],[0,1,translation[1]],[0, 0, 1]])

def applyTransform(crop, transform, camSize):
    '''Applies an affine transform of the crop with range equal to the given camera size'''
    # Requires inverse as the parameters transform from object to camera 
    return crop.transform(camSize, Image.AFFINE, np.linalg.inv(transform).flatten()[:7])

def concatenateTransforms(transforms):
    '''Calculates the resulting transformation in the inverse order'''
    #TODO: Check if transforms[0] pops the first element
    transformDim = transforms[0].shape[0]
    result = np.eye(transformDim)
    for aTransform in transforms:
        result = np.dot(aTransform, result)
    return result

def transform_points(transform, points):
    '''Applies a transformation to an array of homogeneous points'''
    transformedCorners = np.dot(transform, points)
    return transformedCorners
    
#################################
# GENERATION OF COSINE FUNCTIONS
#################################
MIN_AMPLITUDE = 0.2
MAX_AMPLITUDE = 1.2
MIN_PERIOD = 0.25
MAX_PERIOD = 1.0
MIN_PHASE = 0.0
MAX_PHASE = 1.0
MIN_VSHIFT = -0.5
MAX_VSHIFT = 0.5
RANGE = np.arange(0.0, 6.0, 0.1)

def stretch(values, z1, z2):
  mi = min(values)
  ma = max(values)
  return (z2 - z1)*( (values-mi)/(ma-mi) ) + z1

def cosine(y1, y2, randGen):
    a = (MAX_AMPLITUDE - MIN_AMPLITUDE)*randGen.random() + MIN_AMPLITUDE
    b = (MAX_PERIOD - MIN_PERIOD)*randGen.random() + MIN_PERIOD
    c = (MAX_PHASE - MIN_PHASE)*randGen.random() + MIN_PHASE
    d = (MAX_VSHIFT - MIN_VSHIFT)*randGen.random() + MIN_VSHIFT

    f = a*np.cos(b*RANGE - c) + d
    return stretch(f, y1, y2)

#################################
# TRAJECTORY CLASS
#################################

class OffsetTrajectory():

    def __init__(self, w, h, objSize):
        self.thetaMin = -np.pi/12
        self.thetaMax = np.pi/12
        self.xMin = -objSize[0]/2.0
        self.yMin = -objSize[1]/2.0
        self.xMax = objSize[0]/2.0
        self.yMax = objSize[1]/2.0
        self.scaleMax = 1.0
        self.scaleMin = 0.8
        print('Translation bounds: {} to {}'.format([self.xMin, self.yMin], [self.xMax, self.yMax]))
        print('Rotation bounds: {} to {}'.format(self.thetaMin, self.thetaMax))
        print('Scale bounds: {} to {}'.format(self.scaleMin, self.scaleMax))
        self.transforms = [
            #Transformation(translateX, -offset, -offset),
            #Transformation(translateY, -offset, -offset),
            Transformation(rotate, self.thetaMin, self.thetaMax),
            Transformation(scaleX, self.scaleMin, self.scaleMax),
            Transformation(scaleY, self.scaleMin, self.scaleMax),
            Transformation(translateX, self.xMin, self.xMax),
            Transformation(translateY, self.yMin, self.yMax),
        ]

class StretchedSineTrajectoryModel():

  def __init__(self, length):
    self.length = length
    self.randGen = startRandGen()

  def sample(self, sceneSize):
    # Do sampling of starting and ending points (fixed number of steps).
    # Implicitly selects speed, length and direction of movement.
    # Assume constant speed (no acceleration).
    w,h = sceneSize
    x1 = (0.8*w - 0.2*w)*self.randGen.random() + 0.2*w
    y1 = (0.8*h - 0.2*h)*self.randGen.random() + 0.2*h
    x2 = (0.8*w - 0.2*w)*self.randGen.random() + 0.2*w
    y2 = (0.8*h - 0.2*h)*self.randGen.random() + 0.2*h
    logging.debug('Trajectory: from %s,%s to %s,%s',int(x1),int(y1),int(x2),int(y2))

    # Sample direction of waving
    if self.randGen.random() > 0.5:
      # Horizontal steps, vertical wave
      self.X = stretch(RANGE, x1, x2)
      self.Y = cosine(y1, y2, self.randGen)
    else:
      # Horizontal wave, vertical steps
      self.X = cosine(x1, x2, self.randGen)
      self.Y = stretch(RANGE, y1, y2)
    transforms = [
        Transformation(scaleX, 0.9, 1.1),
        Transformation(scaleY, 0.9, 1.1),
        Transformation(translateX, None, None, lambda a,b,steps: self.X),
        Transformation(translateY, None, None, lambda a,b,steps: self.Y),
    ]
    return transforms

#################################
# TRANSFORMATION CLASS
#################################

class Transformation():

  def __init__(self, f, a, b, pathFunction=None, steps=60):
    self.func = f
    self.randGen = startRandGen()
    if pathFunction is None:
        # Initialize range of transformation
        alpha = (b - a)*self.randGen.random() + a
        beta = (b - a)*self.randGen.random() + a
        if alpha > beta:
          c = alpha
          alpha = beta
          beta = c
        # Generate a transformation "path"
        self.X = cosine(alpha, beta, self.randGen)
    else:
        self.X = pathFunction(a, b, steps)

  def transformContent(self, j):
    return self.func(self.X[j])

  def transformShape(self, w, h, j):
    return self.func(w, h, self.X[j])

#################################
# CONTENT TRANSFORMATIONS
#################################

def rotate(angle):
  matrix = applyRotate(angle)
  return matrix

def translateX(value):
  matrix = applyTranslate([value, 0])
  return matrix

def translateY(value):
  matrix = applyTranslate([0, value])
  return matrix

def scaleX(value):
  matrix = applyScale([value, 1])
  return matrix

def scaleY(value):
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

#################################
# OCCLUSSIONS
#################################

class OcclussionGenerator():

  def __init__(self, w, h, maxSize):
    self.randGen = startRandGen()
    num = self.randGen.randint(0,10)
    self.boxes = []
    for i in range(num):
      x1 = (w - maxSize)*self.randGen.random()
      y1 = (h - maxSize)*self.randGen.random()
      wb = maxSize*self.randGen.random()
      hb = maxSize*self.randGen.random()
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

  def __init__(self, sceneFile, objectFile, polygon, maxSegments=9, camSize=(224,224), axes=False, maxSteps=None, contentTransforms=None, shapeTransforms=None, cameraContentTransforms=None, cameraShapeTransforms=None, drawBox=False, camera=True, drawCam=False, trajectoryModel=None, cameraTrajectoryModel=None, trajectoryModelLength=60):
    self.randGen = startRandGen()
    #Number of maximum steps/frames generated
    if maxSteps is None:
        maxSteps = len(RANGE)
    self.maxSteps = maxSteps
    # Load images
    self.scene = Image.open(sceneFile)
    if self.scene.mode == 'L':
        self.scene = self.scene.convert('RGB')
    self.obj = Image.open(objectFile)
    if self.obj.mode == 'L':
        self.obj = self.obj.convert('RGB')
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
    self.polygon = polygon
    self.drawBox = drawBox
    self.drawCam = drawCam
    self.camera = camera
    self.axes = axes
    # Default transformations
    self.shapeTransforms = shapeTransforms
    self.contentTransforms = contentTransforms
    self.trajectoryModelLength = trajectoryModelLength
    if trajectoryModel is not None:
      self.trajectoryModel = TrajectoryModel(trajectoryModel, self.trajectoryModelLength)
    else:
      logging.debug('Random object trajectory model')
      self.trajectoryModel = RandomTrajectoryModel(self.trajectoryModelLength)
      #self.trajectoryModel = StretchedSineTrajectoryModel(self.trajectoryModelLength)
    if cameraTrajectoryModel is not None:
      self.cameraTrajectoryModel = TrajectoryModel(cameraTrajectoryModel, self.trajectoryModelLength)
    else:
      logging.debug('Random camera trajectory model')
      self.cameraTrajectoryModel = RandomTrajectoryModel(self.trajectoryModelLength, step_length_=1e-2)
      #self.cameraTrajectoryModel = StretchedSineTrajectoryModel(self.trajectoryModelLength)
    self.cameraContentTransforms = cameraContentTransforms
    self.cameraShapeTransforms = cameraShapeTransforms
    logging.debug('Scene: %s Object: %s', sceneFile, objectFile)

  def start(self):
    #Segment the object using the polygon and crop to the resulting axes-aligned bounding box
    self.obj = segmentCrop(self.obj, self.polygon)
    # Draw coordinate axes for each source
    if self.axes:
      self.scene = self.draw_axes(self.scene)
      self.obj = self.draw_axes(self.obj)
    self.objSize = self.obj.size
    self.box = [0,0,0,0]
    self.step = 0
    self.validStep = 0
    # Start trajectory
    self.objScaled = self.scaleObject()
    self.objView = self.objScaled
    self.objSize = self.objView.size
    # Calculate bounds after scaling
    self.bounds = np.array([[0,self.objSize[0],self.objSize[0],0],[0,0,self.objSize[1],self.objSize[1]]])
    self.bounds = np.vstack([self.bounds, np.ones((1,self.bounds.shape[1]))])
    self.cameraBounds = np.array([[0,self.camSize[0],self.camSize[0],0],[0,0,self.camSize[1],self.camSize[1]]])
    self.cameraBounds = np.vstack([self.cameraBounds, np.ones((1,self.cameraBounds.shape[1]))])
    self.occluder = OcclussionGenerator(self.scene.size[0], self.scene.size[1], min(self.objSize)*0.3)
    self.currentTransform = np.eye(3,3)
    self.cameraTransform = np.eye(3,3)
    #TODO: reactivate shape transforms
    # Initialize transformations
    if self.shapeTransforms is None:
        self.shapeTransforms = [
            Transformation(identityShape, 1, 1),
        ]
    if self.trajectoryModel is None:
        if self.contentTransforms is None:
            self.contentTransforms = [
                Transformation(scaleX, 0.7, 1.3),
                Transformation(scaleY, 0.7, 1.3),
                Transformation(rotate, -np.pi/50, np.pi/50),
                Transformation(translateX, 0, self.camSize[0]-max(self.objSize)),
                Transformation(translateY, 0, self.camSize[1]-max(self.objSize)),
            ]
        if self.cameraContentTransforms is None:
            #TODO: camera transforms related to sampled trajectory
            cameraDiagonal = np.sqrt(self.camSize[0]**2+self.camSize[1]**2)
            #self.cameraContentTransforms = OffsetTrajectory(self.scene.size[0], self.scene.size[1], cameraDiagonal).transforms
            self.cameraContentTransforms = BoundedTrajectory(self.scene.size[0], self.scene.size[1]).transforms
    else:
        #TODO: Check why the order (object, camera) generates different results
        #Object transform sampling and correction
        self.contentTransforms = self.fit_trajectory(self.bounds, self.camSize, self.trajectoryModel)
        #Camera transform sampling and correction
        self.cameraContentTransforms = self.fit_trajectory(self.cameraBounds, self.scene.size, self.cameraTrajectoryModel)
    if self.cameraShapeTransforms is None:
        self.cameraShapeTransforms = [
            Transformation(identityShape, 1, 1),
        ]
    self.transform()
    self.render()

  def fit_trajectory(self, refBounds, limits, trajectoryModel):
    transformations = trajectoryModel.sample(limits)
    scaleCorr = self.correct_scale(transformations, refBounds, limits)
    transformations += [Transformation(lambda a: scaleCorr, 0,0)]
    transCorr = self.correct_translation(transformations, refBounds, limits)
    transformations += [Transformation(lambda a: transCorr, 0,0)]
    return transformations

  def correct_scale(self, transformations, refBounds, limits):
    logging.debug('Correcting scale with refBounds: %s , limits: %s and %s transformations', refBounds, limits, len(transformations))
    resPoints, resBounds, resSize = self.result_points(transformations, refBounds)
    logging.debug('Resulting bounds: %s Resulting size: %s', resBounds, resSize)
    scaleCorr = 1
    if not np.less(resSize, limits).all():
        ratios = np.array(limits)/resSize
        logging.debug('Ratios: %s', ratios)
        scaleCorr = 0.9*np.min(ratios)
        logging.debug('Scale correction: %s', scaleCorr)
    transCorr = concatenateTransforms((scaleX(scaleCorr), scaleY(scaleCorr)))
    return transCorr

  def result_points(self, transformations, refBounds):
    resPoints = np.array([transform_points(self.transform_step(transformations, i), refBounds) for i in xrange(self.trajectoryModelLength)])
    resBounds = np.array([[np.min(resPoints[:,0,:]), np.min(resPoints[:,1,:])], [np.max(resPoints[:,0,:]), np.max(resPoints[:,1,:])]]).T
    resSize = resBounds[:,1]-resBounds[:,0] 
    return resPoints, resBounds, resSize

  def correct_translation(self, transformations, refBounds, limits):
    logging.debug('Correcting translation with refBounds: %s , limits: %s and %s transformations', refBounds, limits, len(transformations))
    resPoints, resBounds, resSize = self.result_points(transformations, refBounds)
    logging.debug('Resulting bounds: %s Resulting size: %s', resBounds, resSize)
    translation = np.array([[0],[0]])
    if not self.validate_bounds(resBounds, limits):
        gap = limits-resSize
        newPos = [self.randGen.randint(0,int(gap[0])), self.randGen.randint(0,int(gap[1]))]
        translation = np.array(newPos)-resBounds[:2,0]
        logging.debug('Gap: %s New position: %s Translation: %s', gap, newPos, translation)
    transCorr = concatenateTransforms([translateX(translation[0]), translateY(translation[1])])
    return transCorr

  def scaleObject(self):
    # Initial scale of the object is 
    # a fraction of the smallest side of the scene
    smallestSide = min(self.camSize)
    minScale = 0.2
    scaleRange = 0.4
    side = smallestSide*( scaleRange*self.randGen.random() + minScale )
    logging.debug('Scaling object to minimum side: %s', side)
    # Preserve object's aspect ratio with the largest side being "side"
    ar = float(self.obj.size[1])/float(self.obj.size[0])
    if self.obj.size[1] > self.obj.size[0]:
      h = side
      w = side/ar
    else:
      h = side*ar
      w = side
    return self.obj.resize((int(w),int(h)), Image.ANTIALIAS)

  def validate_bounds(self, transformedPoints, size):
    return np.all(np.logical_and(np.greater(transformedPoints[:2,:], [[0], [0]]), np.less(transformedPoints[:2,:], [[size[0]],[size[1]]])))

  def transform_step(self, transformations, step):
    '''Evaluates a step of the transformation for the specified transforms'''
    return concatenateTransforms([transformations[i].transformContent(step) for i in xrange(len(transformations))])

  def transform(self):
    #TODO: reenable shape transforms
    #self.objSize = self.shapeTransforms[0].transformShape(self.objSize[0], self.objSize[1], self.step)
    self.objView = self.objScaled
    # Concatenate transforms and apply them to obtain transformed object
    self.cameraTransform = self.transform_step(self.cameraContentTransforms, self.step)
    self.currentTransform = self.transform_step(self.contentTransforms, self.step)
    self.objView = applyTransform(self.objView, np.dot(self.cameraTransform, self.currentTransform), self.scene.size)

  def render(self):
    self.sceneView = self.scene.copy()
    # Paste the transformed object, but only the axis aligned transformed crop to speed up pasting
    objectTransform = np.dot(self.cameraTransform, self.currentTransform)
    transformedBounds = polygon_bounds(np.dot(objectTransform, self.bounds)[:-1,:].flatten())
    croppedObject = self.objView.crop(transformedBounds)
    self.sceneView.paste(croppedObject, tuple(transformedBounds[:2]), croppedObject)
    self.sceneView = self.occluder.occlude(self.sceneView, self.scene)
    for i in range(len(self.cameraShapeTransforms)):
      self.sceneSize = self.cameraShapeTransforms[i].transformShape(self.scene.size[0], self.scene.size[1], self.step)
      self.sceneView = self.sceneView.resize(self.sceneSize, Image.ANTIALIAS).crop((0,0) + self.scene.size)
    self.camView = applyTransform(self.sceneView, np.linalg.inv(self.cameraTransform), self.camSize)
    # Obtain bounding box points on camera coordinate system
    if self.camera:
        boxPoints = transform_points(self.currentTransform, self.bounds)
        clipSize = self.camSize
    else:
        boxPoints = transform_points(np.dot(self.cameraTransform, self.currentTransform), self.bounds)
        clipSize = self.sceneView.size
    self.box = [max(min(boxPoints[0,:]),0), max(min(boxPoints[1,:]),0), min(max(boxPoints[0,:]), clipSize[0]-1), min(max(boxPoints[1,:]),clipSize[1]-1)]
    self.camDraw = ImageDraw.ImageDraw(self.camView)
    self.sceneDraw = ImageDraw.ImageDraw(self.sceneView)
    if self.drawBox:
        if self.camera:
            self.camDraw.rectangle(self.box)
        else:
            self.sceneDraw.rectangle(self.box)
    if self.drawCam:
        camPoints = transform_points(self.cameraTransform, self.cameraBounds)
        cameraBox = map(int, camPoints[:2, :].T.ravel())
        self.sceneDraw.polygon(cameraBox, outline=(0,255,0))
        sceneBoxPoints = transform_points(np.dot(self.cameraTransform, self.currentTransform), self.bounds)
        objectBox = map(int, sceneBoxPoints[:2, :].T.ravel())
        self.sceneDraw.polygon(objectBox, outline=(0,0,255))
    
  def nextStep(self):
    if self.step < self.maxSteps:
      self.transform()
      self.render()
      self.step += 1
      return True
    else:
      return False

  def saveFrame(self, outDir):
    '''Saves the current frame and appends the bounding box to the ground truth file'''
    fname = os.path.join(outDir, str(self.step).zfill(4) + '.jpg')
    self.getFrame().save(fname)
    gtPath = os.path.join(outDir, 'groundtruth_rect.txt')
    if self.step <= 1:
      out = open(gtPath, 'w')
    else:
      out = open(gtPath, 'a')
    box = map(int,[self.box[0], self.box[1], self.box[2], self.box[3]])
    out.write(','.join(map(str,box)) + '\n' )
    out.close()

  def getFrame(self):
    '''Returns the correct rendered frame'''
    if self.camera:
      return self.camView
    else:
      return self.sceneView

  def getBox(self):
    return self.box

  def convertToGif(self, sequenceDir):
    os.system('convert -delay 1x30 ' + sequenceDir + '/*jpg ' + sequenceDir + '/animation.gif')
    os.system('rm ' + sequenceDir + '*jpg')

  def __iter__(self):
    return self

  def next(self):
    if self.nextStep():
      return self.getFrame()
    else:
      raise StopIteration()

  def draw_axes(self, image):
    '''Draws bottom-left aligned axis for the image reference frame'''
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

def createCOCOSummary(dataDir, dataType, summaryPath):
    try:
        import pycocotools.coco
        annFile = '%s/annotations/instances_%s.json'%(dataDir,dataType)
        #COCO dataset handler object
        print '!!!!!!!!!!!!! WARNING: Loading the COCO annotations can take up to 3 GB RAM !!!!!!!!!!!!!'
        coco = pycocotools.coco.COCO(annFile)
        #TODO: Filter the categories to use in sequence generation
        catIds = coco.getCatIds()
        cats = coco.loadCats(catIds)
        catDict = {cat['id']:cat['name'] for cat in cats}
        #Ommitted category filter as it seems to return less results/images
        imgIds = coco.getImgIds()
        objData = coco.loadImgs(imgIds)
        objAnnIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
        objAnns = coco.loadAnns(objAnnIds)
        objFileNames = {obj['id']:obj['file_name'] for obj in objData}
        cocoDict = [
            {
                k:obj.get(k, objFileNames[obj['image_id']] if k == 'file_name' else None)
                for k in ['category_id', 'image_id', 'segmentation', 'file_name']
            }
            for obj in objAnns
        ]
        print 'Number of categories {} and corresponding images {}'.format(len(catIds), len(imgIds))
        print 'Category names: {}'.format(', '.join(catDict.values()))
        summary = {SUMMARY_KEY: cocoDict, CATEGORY_KEY: catDict}
        summaryFile = open(summaryPath, 'w')
        pickle.dump(summary, summaryFile)
        summaryFile.close()
        #Free memory
        del coco, catIds, cats, catDict, imgIds, objData, objAnnIds, objAnns
    except ImportError as e:
        print 'No support for pycoco'
        raise e

SUMMARY_KEY='summary'
CATEGORY_KEY='categories'

class SimulatorFactory():

    def __init__(self, dataDir, trajectoryModelPath, summaryPath, scenePathTemplate = 'images/train2014', objectPathTemplate = 'images/train2014'):
        self.dataDir = dataDir
        self.scenePathTemplate = scenePathTemplate
        self.objectPathTemplate = objectPathTemplate
        print 'Loading summary from file {}'.format(summaryPath)
        summaryFile = open(summaryPath, 'r')
        self.summary = pickle.load(summaryFile)
        summaryFile.close()
        #Default model is Random, overwrite if specified
        self.trajectoryModel = None
        if trajectoryModelPath is not None:
            modelFile = open(trajectoryModelPath, 'r')
            self.trajectoryModel = pickle.load(modelFile)
            modelFile.close()
        self.sceneList = os.listdir(os.path.join(self.dataDir, self.scenePathTemplate))

    def createInstance(self, *args, **kwargs):
        '''Generates TrajectorySimulator instances with a random scene from the scene template and a random object from the object template'''
        self.randGen = startRandGen()
        #Select a random image for the scene
        scenePath = os.path.join(self.dataDir, self.scenePathTemplate, self.randGen.choice(self.sceneList))

        #Select a random image for the object
        objData = self.randGen.choice(self.summary[SUMMARY_KEY])
        objPath = os.path.join(self.dataDir, self.objectPathTemplate, objData['file_name'].strip())

        #Select a random object in the scene and read the segmentation polygon
        logging.debug('Segmenting object from category %s', self.summary[CATEGORY_KEY][int(objData['category_id'])])
        polygon = self.randGen.choice(objData['segmentation'])

        simulator = TrajectorySimulator(scenePath, objPath, polygon=polygon, trajectoryModel=self.trajectoryModel, cameraTrajectoryModel=self.trajectoryModel, *args, **kwargs)
        
        return simulator

class RandomTrajectoryModel(object):
    #TODO: multiple objects
    #TODO: different models for background and object, specifically different acc value

    def __init__(self, length, step_length_=0.1, vel_scale=1, acc_scale=0.1, scale_range=0.1):
        self.length = length
        self.step_length_ = step_length_
        self.vel_scale = vel_scale
        self.acc_scale = acc_scale
        self.scale_range = scale_range

    def sample(self, sceneSize):
        tx, ty, sx, sy = self.getRandomTrajectory(sceneSize)

        transforms = [
            Transformation(scaleX, None, None, pathFunction=lambda a,b,steps: sx),
            Transformation(scaleY, None, None, pathFunction=lambda a,b,steps: sy),
            Transformation(translateX, None, None, pathFunction=lambda a,b,steps: tx),
            Transformation(translateY, None, None, pathFunction=lambda a,b,steps: ty),
        ]
        return transforms

    def getRandomTrajectory(self, sceneSize):
        # Initial position uniform random inside the box.
        y = np.random.rand()
        x = np.random.rand()

        # Choose a random velocity.
        theta = np.random.rand() * 2 * np.pi
        start_vel = np.random.normal(0, self.vel_scale)
        v_y = start_vel * np.sin(theta)
        v_x = start_vel * np.cos(theta)

        start_y = np.zeros((self.length,))
        start_x = np.zeros((self.length,))
        scales = np.zeros((self.length,))
        for i in range(self.length):
            scales[i] = np.exp((np.random.random_sample()-0.5)*self.scale_range)
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            v_y += 0 if self.acc_scale == 0 else np.random.normal(0, self.acc_scale)
            v_x += 0 if self.acc_scale == 0 else np.random.normal(0, self.acc_scale)

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_x = (sceneSize[0] * start_x).astype(np.int32)
        start_y = (sceneSize[1] * start_y).astype(np.int32)
        return start_x, start_y, scales, scales
        

class GMMTrajectoryModel():

    def __init__(self, model, length, maxSize=10, base=10.0):
        self.model = model
        self.length = length
        self.maxSize = maxSize

    def sample(self, sceneSize):
        return self.getAveragedTrajectory(sceneSize)

    def getAveragedTrajectory(self, sceneSize, base=10.0):
        '''Generates a sample trajectory from the model by taking the average of a random sample of clusters'''
        self.randGen = startRandGen()
        nComponents = self.model.n_components
        clusterIds = np.array([self.randGen.randrange(nComponents) for i in np.arange(self.randGen.randrange(1, self.maxSize))])
        #Take the mean of the sampled clusters and reshape them using the length attribute and split into parameters
        trajectory = np.mean(self.model.means_[clusterIds], axis=0)
        trajectory = trajectory.reshape(int(trajectory.shape[0]/self.length), self.length)
        tx, ty, sx, sy = trajectory
        #Integrate if model generates relative parameters
        if self.model.relative:
            tx0, ty0 = (self.randGen.random()*sceneSize[0], self.randGen.random()*sceneSize[1])
            tx = np.cumsum(tx)
            ty = np.cumsum(ty)
            sx = np.cumsum(sx)
            sy = np.cumsum(sy)
        else:
            tx0, ty0 = (0,0)
        #Revert normalization
        #TODO: improve denormalization by extracting de/normalization to a class
        tx = tx*sceneSize[0]+tx0
        ty = ty*sceneSize[1]+ty0
        sx = base**sx
        sy = base**sy
        transforms = [
            Transformation(scaleX, None, None, pathFunction=lambda a,b,steps: sx),
            Transformation(scaleY, None, None, pathFunction=lambda a,b,steps: sy),
            Transformation(translateX, None, None, pathFunction=lambda a,b,steps: tx),
            Transformation(translateY, None, None, pathFunction=lambda a,b,steps: ty),
        ]
        return transforms

class SineTrajectoryModel():
    def __init__(self, length):
        self.length = length

    def sample(self, sceneSize):
        transforms = [
            Transformation(scaleX, 0.9, 1.1, None, steps=self.length),
            Transformation(scaleY, 0.9, 1.1, None, steps=self.length),
            Transformation(translateX, 0, sceneSize[0], None, steps=self.length),
            Transformation(translateY, 0, sceneSize[1], None, steps=self.length),
        ]
        return transforms

class TrajectoryModel(object):
    def __init__(self, model, length):
        self.length = length
        self.gmmModel = GMMTrajectoryModel(model, self.length)
        self.randomModel = RandomTrajectoryModel(self.length)
        self.sineModel = SineTrajectoryModel(self.length)
        self.stretchedModel = StretchedSineTrajectoryModel(self.length)
        self.models = [self.randomModel, self.sineModel, self.stretchedModel, self.gmmModel]
        self.randGen = startRandGen()

    def sample(self, sceneSize):
        modelIndex = self.randGen.randrange(len(self.models))
        logging.debug('Sampling trajectory from model: %s', self.models[modelIndex])
        return self.models[modelIndex].sample(sceneSize)
