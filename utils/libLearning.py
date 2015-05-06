import numpy as np
import utils as cu
import libDetection as det
import dataProcessor as dp
from utils import emptyMatrix

###############################################
# Hard Negative Mining
###############################################

class HardNegativeMining():
  def __init__(self,currentModel,maxVectorsPerImage):
    self.currentModel = currentModel
    self.maxVectorsPerImage = maxVectorsPerImage

  def run(self,img,features,bboxes):
    pred = self.currentModel.predict(features,bboxes)
    candidates = pred > -1.0001
    f = features[candidates]
    p = pred[candidates]
    bboxes = [bboxes[i] for i in range(len(bboxes)) if candidates[i]]
    # Sort candidate detections by score
    s = np.argsort(p)
    j = min(2*self.maxVectorsPerImage,f.shape[0])
    # Keep top candidates only
    if j > 0:
      return (f[ s[-j:] ], p[ s[-j:] ], bboxes)
    else:
      return None

def getHardNegatives(negativesDir,negativesList,featuresExt,numFeatures,maxVectors,currentModel):
  maxVectorsPerImage = maxVectors/len(negativesList)
  i = 0
  task = HardNegativeMining(currentModel,maxVectorsPerImage)
  result = dp.processData(negativesList,negativesDir,featuresExt,task)
  hardng = emptyMatrix([2*maxVectors,numFeatures])
  boxes = []
  while len(result) > 0:
    data = result.pop(0)
    if data[0].shape[0]+i > hardng.shape[0]:
      print 'Not enough matrix space'
      hardng = np.concatenate( (hardng,emptyMatrix([maxVectors,numFeatures])) )
    hardng[i:i+data[0].shape[0],:] = data[0]
    boxes += data[2]
    i = i + data[0].shape[0]
  return hardng[0:i,:],boxes[0:i]
  
###############################################
# Random Negative Windows Filter
###############################################

class RandomNegativesFilter():
  def __init__(self,numFeatures,randomBoxes):
    self.numFeatures = numFeatures
    self.randomBoxes = randomBoxes

  def run(self,img,features,bboxes):
    boxes = range(0,features.shape[0])
    cu.rnd.shuffle(boxes)
    m = min(features.shape[0],self.randomBoxes)
    bboxes = [bboxes[i] for i in boxes]
    return (features[boxes[0:m]],bboxes)

def getRandomNegs(featuresDir,negativeList,featuresExt,numFeatures,maxVectors,maxNegativeImages):
  randomBoxes = maxVectors/maxNegativeImages
  cu.rnd.shuffle(negativeList)
  task = RandomNegativesFilter(numFeatures,randomBoxes)
  negatives = [negativeList.pop(0) for i in range(maxNegativeImages)]
  result = dp.processData(negatives,featuresDir,featuresExt,task)
  neg = emptyMatrix([maxVectors,numFeatures])
  boxes = []
  n = 0
  while len(result) > 0:
    mat,box = result.pop()
    neg[n:n+mat.shape[0]] = mat
    n = n + mat.shape[0]
    boxes += box
  return (neg[0:n],boxes[0:n])

###############################################
# Negative-Windows-From-Positive-Images Filter
###############################################
class NWFPIFilter():
  def __init__(self,groundTruths,featuresDir,featuresExt,maxNegatives,overlap,model):
    self.groundTruths = groundTruths
    self.featuresDir = featuresDir
    self.featuresExt = featuresExt
    self.maxNegatives = maxNegatives
    self.overlap = overlap
    self.model = model

  def rank(self,img,features,bboxes):
    pred = self.model.predict(features,bboxes)
    candidates = pred > -1.0001
    f = features[candidates]
    b = [bboxes[t] for t in range(len(bboxes)) if candidates[t]]
    p = pred[candidates]
    # Sort candidate detections by score
    s = np.argsort(p)
    j = min(2*self.maxNegatives,f.shape[0])
    # Keep top candidates only
    if j > 0:
      return (f[ s[-j:] ], [ b[t] for t in s[-j:] ])
    else:
      return None,None

  def run(self,img,features,bboxes):
    if self.model:
      features,bboxes = self.rank(img,features,bboxes)
    if features == None:
      return ([],[],[],[])
    positives,negatives = [],[]
    imageData = self.groundTruths[img]
    for i in range( len(bboxes) ):
      isPositive,isNegative = False,False
      for j in imageData:
        o = det.IoU(j,map(float,bboxes[i][1:]))
        if o >= 0.85:
          isPositive = True
          break
        elif self.overlap >= o and o > 0:
          isNegative = True
      if isPositive: 
        positives.append(i)
      if isNegative:
        negatives.append(i)
    if self.model:
      negatives.reverse()
    else:
      cu.rnd.shuffle(negatives)
    posIdx = [bboxes[t] for t in positives]
    posFeat = [features[positives]]
    negIdx = [bboxes[t] for t in negatives[0:self.maxNegatives]]
    negFeat = [features[negatives[0:self.maxNegatives]]]
    return (posIdx,posFeat,negIdx,negFeat)

def selectNegativeWindowsFromPositiveImages(groundTruths,featuresDir,featuresExt,maxVectors,overlap,model=False):
  gtb = dict()
  for x in groundTruths:
    im,bx = x[0],map(float,x[1:])
    try:
      gtb[im].append(bx)
    except:
      gtb[im] = [bx]

  task = NWFPIFilter(gtb,featuresDir,featuresExt,maxVectors/len(gtb.keys()),overlap,model)
  result = dp.processData(gtb.keys(),featuresDir,featuresExt,task)
  posIdx,posFeat,negIdx,negFeat = [],[],[],[]
  for r in result:
    posIdx  += r[0]
    posFeat += r[1]
    negIdx  += r[2]
    negFeat += r[3]
  Xp = emptyMatrix( (len(posIdx),posFeat[0].shape[1]) )
  Xn = emptyMatrix( (len(negIdx),negFeat[0].shape[1]) )
  k = 0
  for i in range(len(posFeat)):
    Xp[k:k+posFeat[i].shape[0],:] = posFeat[i]
    k = k + posFeat[i].shape[0]
  k = 0
  for i in range(len(negFeat)):
    Xn[k:k+negFeat[i].shape[0],:] = negFeat[i]
    k + k + negFeat[i].shape[0]

  print 'NegFromPos ready:',len(negIdx)
  
  return {'posIdx':posIdx, 'posFeat':Xp, 'negIdx':negIdx, 'negFeat':Xn} 

###############################################
# Cross-validation evaluation
###############################################

def reportCrossValidationPerformance(clf,X,Y):
  from sklearn import cross_validation
  import sklearn.metrics as met
  skf = cross_validation.StratifiedKFold(Y, n_folds=10)
  p,r = 0.0,0.0
  for train_index, test_index in skf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clf.fit(X_train,Y_train)
    # Performance measures:
    pred = clf.predict(X_test)
    cfm = met.confusion_matrix(Y_test,pred)
    precision = float(cfm[1][1])/(cfm[1][1] + cfm[0][1])
    recall = float(cfm[1][1])/(cfm[1][1] + cfm[1][0])
    p += precision
    r += recall
    print '{:.4f} {:.4f}'.format(precision,recall)
    print cfm
  print 'AVG {:.4f} {:.4f}'.format(p/10.0, r/10.0)

###############################################
# Load Hard Negatives from predefined list
###############################################

class LoadHardNegatives():
  def __init__(self,boxInfo):
    self.boxInfo = boxInfo

  def run(self,img,features,bboxes):
    wanted = set([ ':'.join(map(str,x)) for x in self.boxInfo[img] ])
    candidates = []
    imgList = []
    box = []
    for i in range(len(bboxes)):
      b = bboxes[i]
      boxHash = ':'.join(b[1:])
      if boxHash in wanted:
        candidates.append(True)
        imgList.append(img)
        box.append(b)
        wanted.remove(boxHash)
      else:
        candidates.append(False)
    candidates = np.asarray(candidates)
    return (features[candidates],imgList,box)

def loadHardNegativesFromList(featuresDir,negativesInfo,featuresExt,numFeatures,totalNegatives,idx=False):
  i = 0
  task = LoadHardNegatives(negativesInfo)
  result = dp.processData(negativesInfo.keys(),featuresDir,featuresExt,task)
  hardng = emptyMatrix([totalNegatives,numFeatures])
  hardNames = []
  boxes = []
  while len(result) > 0:
    data,imgs,box = result.pop(0)
    hardng[i:i+data.shape[0],:] = data
    hardNames += imgs
    boxes += box
    i = i + data.shape[0]
  return (hardng[0:i,:],boxes)

def parseRankedDetectionsFile(detectionsLog,maxNegOverlap,maxNegativeVectors):
  ## Read ranked list of negatives
  if isinstance(detectionsLog, basestring):
    log = [x.split() for x in open(detectionsLog)]
  else:
    log = detectionsLog
  posExamples = dict()
  negExamples = dict()
  posCount,negCount,noCares,negTaken = 0,0,0,0
  for l in log:
    if l[7] == '1':
      posCount += 1
      try:
        posExamples[l[0]] += [ l[1:5] ]
      except:
        posExamples[l[0]] = [ l[1:5] ]
    elif l[7] == '0' and float(l[6]) <= maxNegOverlap:
      negCount += 1
      if negCount < maxNegativeVectors:
        negTaken += 1
        try:
          negExamples[l[0]] += [ l[1:5] ]
        except:
          negExamples[l[0]] = [ l[1:5] ]
    else:
      noCares += 1
  print 'NEGEXAMPLES:',np.sum( [len(negExamples[i]) for i in negExamples.keys()] )
  print 'Log Of Detections: Pos {:} Neg {:} NoCares {:}'.format(posCount,negCount,noCares)
  return {'posExamples':posExamples,'negExamples':negExamples,'negTaken':negTaken}

###############################################
# Compute aspect ratio and size features
###############################################

def addAspectRatioAndSizeFeatures(features,index,aspectRatios,objectSizes):
  fs = 12
  boxes = np.asmatrix( [ map(float,x[1:]) for x in index] )
  imgSize = np.max(boxes,axis=0)[0,2:]
  sizes = np.asarray(boxes[:,2]-boxes[:,0])*np.asarray(boxes[:,3]-boxes[:,1])/(500*500) #/(imgSize[0,0]*imgSize[0,1])
  sizeF = np.tile(sizes, (1, fs)) - np.tile(objectSizes,(sizes.shape[0],1))
  ratios = np.asarray(boxes[:,2]-boxes[:,0])/np.asarray(boxes[:,3]-boxes[:,1])
  ratioF = np.tile(ratios, (1, fs)) - np.tile(aspectRatios,(sizes.shape[0],1))
  # **
  S = np.argsort(np.abs(sizeF),axis=1)
  R = np.argsort(np.abs(ratioF),axis=1)
  for i in range(len(S)):
    sizeF[ i, S[i] ] = [1.,.5,.25,.125,.0625,0.,0.,0.,0.,0.,0.,0.]
    ratioF[i, R[i] ] = [1.,.5,.25,.125,.0625,0.,0.,0.,0.,0.,0.,0.]
  # **
  return np.concatenate( (features, sizeF, ratioF), axis=1 )

def computeAspectRatioAndSizeIntervals(index):
  boxes = np.asmatrix( [ map(float,x[1:]) for x in index] )
  imgSize = np.max(boxes,axis=0)[0,2:]
  sizes = np.asarray(boxes[:,2]-boxes[:,0])*np.asarray(boxes[:,3]-boxes[:,1])/(500*500) #/(imgSize[0,0]*imgSize[0,1])
  objectSizes = np.percentile(sizes,range(5,100,10))
  ratios = np.asarray(boxes[:,2]-boxes[:,0])/np.asarray(boxes[:,3]-boxes[:,1])
  aspectRatios = np.percentile(ratios,range(5,100,10))
  # **
  objectSizes = [objectSizes[0]*0.5] + objectSizes + [1.0]
  aspectRatios = [aspectRatios[0]*0.5] + aspectRatios + [aspectRatios[-1]*1.5]
  # **
  print 'AspectRatios:',aspectRatios
  print 'ObjectSizes:',objectSizes
  return aspectRatios,objectSizes

