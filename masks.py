import os,sys
import libDetection as det
import numpy as np
import matplotlib.pyplot as plt
import Image
import utils as cu
import warp as wp

def loadArchitecture(filename):
  arch = [x.split() for x in open(filename)]
  for row in arch:
    row[1:] = map(int,row[1:])
  return arch

def getLayerIndex(arch,layerName):
  for i in range(len(arch)):
    if arch[i][0] == layerName:
      return i
  return -1

def loadBoxIndexFile(filename):
  gt = [x.split() for x in open(filename)]
  images = {}
  for k in gt:
    try:
      images[k[0]] += [ map(float,k[1:]) ]
    except:
      images[k[0]] = [ map(float,k[1:]) ]
  return images

###########################################
## PROJECTION METHOD
###########################################
def projectCoordinateToPreviousLayer(arch,lidx,x,y,w,h):
  if lidx > 0:
    if x < arch[lidx][2] and y < arch[lidx][3]:
      wp = w*arch[lidx][4] - (arch[lidx][4]-arch[lidx][5])*(w - 1)
      hp = h*arch[lidx][4] - (arch[lidx][4]-arch[lidx][5])*(w - 1)
      return (x*arch[lidx][5],y*arch[lidx][5],wp,hp)
    else:
      print 'Coordinates out of plane in layer',layer,':',x,y
  else:
    return (x,y,w,h)

def projectToImagePlane(arch,layer,x,y,w,h):
  lidx = getLayerIndex(arch,layer)
  while lidx > 0:
    x,y,w,h = projectCoordinateToPreviousLayer(arch,lidx,x,y,w,h)
    #print arch[lidx-1][0],x,y,w,h
    lidx -= 1
  x2,y2 = min(arch[lidx][2],x+w),min(arch[lidx][2],y+h)
  w,h = x2-x,y2-y
  return x,y,x2,y2,w,h

def projectAllData(arch,fromLayer):
  lidx = getLayerIndex(arch,fromLayer)
  print arch[0][0]
  for i in range(arch[lidx][2]):
   for j in range(arch[lidx][2]):
  #   #print i,j,projectCoordinateToPreviousLayer(arch,lidx,i,j)
      print i,j,projectToImagePlane(arch,fromLayer,i,j,1,1)
  #print projectToImagePlane(arch,layer,5,5,1,1)

###########################################
## CONVOLUTION METHOD
###########################################
def getCoordMatrix(size):
  M = []
  for i in range(size):
    M.append([])
    for j in range(size):
      M[i].append( (j,i,j,i) )
  return M

def convolve(M,filterSize,stride):
  R = []
  if stride == 1:
    outputSize = len(M)
  else:
    outputSize = (len(M)-filterSize)/stride + 1
  x,y = -stride,-stride
  for i in range(outputSize): #rows
    y += stride
    x = -stride
    R.append([])
    for j in range(outputSize):#cols
      x += stride
      #print i,j,x,y
      R[i].append([])
      for m in range(filterSize):
        for n in range(filterSize):
          if x+n < len(M) and y+m < len(M):
            R[i][j] += [M[y+m][x+n]]
  return R
      
def findAreas(R):
  s = len(R)
  A = []
  for i in range(s):
    A.append([])
    for j in range(s):
      c1 = reduce(lambda x,y: x if x[0]+x[1]<y[0]+y[1] else y, R[i][j])
      c2 = reduce(lambda x,y: x if x[2]+x[3]>y[2]+y[3] else y, R[i][j])
      A[i].append( c1[0:2]+c2[2:4] )
  return A

def projectCoordsToReceptiveField(arch,fromLayer):
  lidx = getLayerIndex(arch,fromLayer)
  data = getCoordMatrix(arch[0][2])
  for layer in range(1,lidx+1):
    R = convolve(data,arch[layer][4],arch[layer][5])
    A = findAreas(R)
    data = A
  return data

###########################################
## COORDINATE PROJECTION FUNCTIONS
###########################################
def intersectWithGroundTruth(A,gt):
  s = len(A)
  R = np.zeros((s,s))
  for i in range(s):
    for j in range(s):
      maxIou,maxOv,maxCov = 0.,0.,[0.,[]]
      for k in gt:
        iou = det.IoU( A[i][j], k )
        ov = det.overlap( k,A[i][j] )
        cov = det.overlap( A[i][j], k )
        if iou > maxIou: maxIou = iou
        if ov > maxOv: maxOv = ov
        if cov > maxCov[0]: maxCov = [cov,k]
      #R[i,j] = maxOv
      #continue
      if maxIou >= 0.7: # Relative size is roughly the same
        R[i,j] = 1.
      #elif maxOv >= 0.7: # The object covers this area almost completely
      #  R[i,j] = 1.
      elif maxCov[0] >= 0.7: # The object is covered by the area
        ox = maxCov[1][0] + (maxCov[1][2]-maxCov[1][0])/2.
        oy = maxCov[1][1] + (maxCov[1][3]-maxCov[1][1])/2.
        aw = A[i][j][2] - A[i][j][0]
        ah = A[i][j][3] - A[i][j][1]
        ax = A[i][j][0] + aw/2.
        ay = A[i][j][1] + ah/2.
        r = 0.2
        if (ax-r*aw <= ox and ox <= ax+r*aw) and (ay-r*ah <= oy and oy <= ay+r*ah):
          R[i,j] = 1.
  return R

def projectScoresBackToImagePlane(S,A):
  s = len(S)
  t = max(max(max(A)))+1
  P = np.zeros((t,t))
  T = np.ones((t,t))
  for i in range(s):
    for j in range(s):
      a = A[i][j]
      P[a[1]:a[3],a[0]:a[2]] += S[i][j]
      T[a[1]:a[3],a[0]:a[2]] += 1.
  P = P/T
  P[P<0.5] = 0.
  return P
  
def projectFeatureMapToImagePlane(target,F):
    wfunc = wp.Warping()
    s = len(F)-1
    box = [ F[0][0][0],F[0][0][1],F[s][s][2],F[s][s][3] ]
    wfunc.prepare(box,target)
    P = []
    for i in range(s+1):
        P.append([])
        for j in range(s+1):
            P[i].append( wfunc.transform(F[i][j]) )
    return P

def visualizeGridAndGroundTruths(image, boxes, groundTruth, scaleTo, maps):
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  from matplotlib.patches import Rectangle
  import Image

  fig = plt.figure()
  img = mpimg.imread(image)
  (ox,oy,oc) = img.shape
  bIm = Image.fromarray(np.uint8(img))
  img = np.asarray(bIm.resize((scaleTo[0],scaleTo[1]), Image.ANTIALIAS))
  (X,Y,C) = img.shape
  a = fig.add_subplot(1,3,1)
  imgplot = plt.imshow(img,origin='lower')
  currentAxis = plt.gca()
  alph = 0.2
  for i in range(len(boxes)):
    boxes[i] = [boxes[i][0]] + map(float,boxes[i][1:])
  # Draw boxes 
  for j in range(len(boxes)):
    b = map(float,boxes[j][0:])
    currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=1.0, fill=False, color=(1,1,1),alpha=alph))
    if j > 20: break
  # Draw ground truths
  for gt in groundTruth:
    a = rescaleBox( map(float,gt), scaleTo[0]/float(oy), scaleTo[1]/float(ox))
    print gt,a,ox,oy,X,Y
    currentAxis.add_patch(Rectangle((a[0], X-a[1]), a[2]-a[0], a[1]-a[3], linewidth=0.0, fill=True, color=(0,1,0),alpha=0.3))

  R,P,A = maps
  a = fig.add_subplot(1,3,2)
  imgplot = plt.imshow(R)
  a = fig.add_subplot(1,3,3)
  imgplot = plt.imshow(P)
  plt.show()


###########################################
## OTHER AUXILIARY FUNCTIONS
###########################################
def listMappings(A):
  s = len(A)
  map = []
  for i in range(s):
    for j in range(s):
      map.append( (i,j)+A[i][j] )
  return map

def listBoxes(A):
  map = listMappings(A)
  return [x[2:] for x in map]

def rescaleBox(box,xFactor,yFactor):
  return (xFactor*box[0],yFactor*box[1],xFactor*box[2],yFactor*box[3])

def rescaleAllBoxes(boxes,xf,yf):
  return [rescaleBox(box,xf,yf) for box in boxes]

def rescaleBoxesInMatrix(A,xf,yf):
  return [rescaleAllBoxes(boxes,xf,yf) for boxes in A]

def testProcedure():
  imgsDir = '/home/caicedo/data/allimgs/'
  arch = loadArchitecture('cnn.arch')
  A = projectCoordsToReceptiveField(arch,'conv3')
  boxes = listBoxes(A)
  gt = loadBoxIndexFile('/home/caicedo/data/rcnn/lists/2007/trainval/aeroplane_gt_bboxes.txt')
  for imName in gt.keys():
    imgFile = imgsDir+'/'+imName+'.jpg'
    w,h = Image.open(imgFile).size
    R = intersectWithGroundTruth(A,rescaleAllBoxes(gt[imName],227./w, 227./h))
    #print R
    P = projectScoresBackToImagePlane(R,A)
    visualizeGridAndGroundTruths(imgFile, boxes, gt[imName], (227,227),(R,P,A))
    break

def computeGlobalImageMask():
  params = cu.loadParams('category imgsDir groundTruthsFile layer outputDir')
  arch = loadArchitecture('cnn.arch')
  A = projectCoordsToReceptiveField(arch,params['layer'])
  boxes = listBoxes(A)
  gt = loadBoxIndexFile(params['groundTruthsFile'])
  for imName in gt.keys():
    imgFile = params['imgsDir']+'/'+imName+'.jpg'
    w,h = Image.open(imgFile).size
    R = intersectWithGroundTruth(A,rescaleAllBoxes(gt[imName],227./w, 227./h))
    cu.saveMatrix(R,params['outputDir']+'/'+imName+'.'+params['category'])

def multipleRegionMasks():
  params = cu.loadParams('category imgsDir groundTruthsFile layer featuresDir outputDir')
  arch = loadArchitecture('cnn.arch')
  A = projectCoordsToReceptiveField(arch,params['layer'])
  s = len(A)
  gt = loadBoxIndexFile(params['groundTruthsFile'])
  for imName in gt.keys():
    imgFile = params['imgsDir']+'/'+imName+'.jpg'
    w,h = Image.open(imgFile).size
    idx = loadBoxIndexFile(params['featuresDir'] + '/' + imName + '.idx')
    M = np.zeros((len(idx[imName]),s,s))
    i = 0
    for box in idx[imName]:
        P = projectFeatureMapToImagePlane(box,A)
        M[i,:,:] = intersectWithGroundTruth(P,gt[imName])
        i += 1
    cu.saveMatrix(M,params['outputDir']+'/'+imName+'.'+params['category'])

if __name__ == '__main__':
  multipleRegionMasks()

