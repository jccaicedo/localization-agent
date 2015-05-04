import os,sys
import numpy as np
from abc import ABCMeta, abstractmethod

intersect = lambda x,y: [max(x[0],y[0]),max(x[1],y[1]),min(x[2],y[2]),min(x[3],y[3])]

area = lambda x: (x[2]-x[0]+1)*(x[3]-x[1]+1)

# Symmetric Jaccard coefficient
def IoU(b1,b2):
  bi = intersect(b1,b2)
  iw = bi[2] - bi[0] + 1
  ih = bi[3] - bi[1] + 1
  if iw > 0 and ih > 0:
    ua = area(b1) + area(b2) - iw*ih
    overlap = iw*ih/ua
    return overlap
  else:
    return 0

# How much box1 covers box2
def overlap(b1,b2):
  bi = intersect(b1,b2)
  iw = bi[2] - bi[0] + 1
  ih = bi[3] - bi[1] + 1
  if iw > 0 and ih > 0:
    #ua = area(b1) + area(b2) - iw*ih
    overlap = iw*ih/float(area(b2))
    return overlap
  else:
    return 0

def nonMaximumSuppression(boxes,scores,maxOverlap):
  if maxOverlap == 1.0:
    return [map(float,x[1:]) for x in boxes],scores
  s = np.argsort(scores)
  if len(boxes[0]) > 4:
    boxes = [map(float,boxes[t][1:]) for t in s]
  elif len(boxes[0]) == 4:
    boxes = [boxes[t] for t in s]
  scores = [scores[t] for t in s]
  filteredBoxes = []
  filteredScores = []
  while len(boxes) > 0:
    prevBox = boxes.pop()
    filteredBoxes.append(prevBox)
    filteredScores.append(scores.pop())
    suppressed = []
    for j in range(len(boxes)-1,-1,-1):
      box = boxes[j]
      ov = IoU(prevBox,box)
      if ov > maxOverlap:
        suppressed.append(j)
    for sprs in suppressed:
      boxes.pop(sprs)
      scores.pop(sprs)
  return (filteredBoxes,filteredScores)

########################################
## ABSTRACT DETECTOR
########################################
class Detector(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def learn(self,pos,neg,posIdx,negIdx):
    pass

  @abstractmethod
  def predict(self,X,Z):
    pass

  @abstractmethod
  def predictAll(self,X,Z):
    pass

  @abstractmethod
  def load(self,modelFile):
    pass

  @abstractmethod
  def save(self,modelFile):
    pass

########################################
## DETECTOR FACTORY
########################################
def createDetector(modelType,params=None):
  if modelType == 'latent':
    import latentSVM as lsvm
    detector = lsvm.LatentSVM(params)
  elif modelType == 'linear':
    import linearDetector as linear
    detector = linear.LinearDetector(params)
  elif modelType == 'single':
    import subcatDetector as subcat
    detector = subcat.SingleDetector(params)
  elif modelType == 'subcategories':
    import subcatDetector as subcat
    detector = subcat.SubcategoriesDetector(params)
  else:
    print 'Model not supported'
    return None
  return detector

########################################
## OTHER AUXILIARY FUNCTIONS
########################################
def showDetections(image, boxes, scores, fill=False, outputFile=None):
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  from matplotlib.patches import Rectangle

  img=mpimg.imread(image)
  (X,Y,C) = img.shape
  plt.clf()
  imgplot = plt.imshow(img,origin='lower')
  maxScore = max(scores)
  minScore = min(scores)
  currentAxis = plt.gca()
  alph = 2.0/float(len(boxes)) if len(boxes) > 1 else 0.1
  s = np.argsort(scores)
  for j in range(len(boxes)):
    i = s[j]
    b = map(float,boxes[i][0:])
    z = (scores[i]-minScore)/(maxScore-minScore) if maxScore != minScore else 1.0
    c = (1-z,0,z)
    #c = (0,1,0)
    if fill:
      currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=0.0, fill=True, color=c,alpha=alph))
    #currentAxis.annotate("{:10.2f}".format(scores[i]), xy=(b[0],X-b[1]),color='white')
    currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=1.0, fill=False, color=str(z)))
    #currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=3.0, fill=False, color=(1,0,0),alpha=alph))
  if outputFile == None:
    plt.show()
  else:
    plt.savefig(outputFile, bbox_inches='tight')

def showObjectLayout(image, boxes, scores, fill=False, outputFile=None):
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  from matplotlib.patches import Rectangle

  img=mpimg.imread(image)
  (X,Y,C) = img.shape
  plt.clf()
  imgplot = plt.imshow(img,origin='lower')
  currentAxis = plt.gca()
  alph = 2.0/float(len(boxes)) if len(boxes) > 1 else 0.1
  colors = [ "#00FF00", "#FF0000" ] + [ "#0000FF" for i in range(len(boxes) - 2) ]
  for j in range(len(boxes)):
    b = map(float,boxes[j][0:])
    if fill:
      currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=0.0, fill=True, color=c,alpha=alph))
    #currentAxis.annotate("{:10.2f}".format(scores[j]), xy=(b[0],X-b[1]),color='white')
    currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=2.0, fill=False, color=colors[j]))
  if outputFile == None:
    plt.show()
  else:
    plt.savefig(outputFile, bbox_inches='tight')


def showBestMatches(image, boxes, scores, groundTruth):
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  from matplotlib.patches import Rectangle

  img=mpimg.imread(image)
  (X,Y,C) = img.shape
  plt.clf()
  imgplot = plt.imshow(img,origin='lower')
  maxScore = max(scores)
  minScore = min(scores)
  currentAxis = plt.gca()
  alph = 0.2
  for i in range(len(boxes)):
    boxes[i] = [boxes[i][0]] + map(float,boxes[i][1:])

  # Draw top scoring proposals
  s = np.argsort(scores)
  for j in range(len(boxes)):
    i = s[j]
    b = map(float,boxes[i][0:])
    z = (scores[i]-minScore)/(maxScore-minScore)
    c = (1-z,0,z)
    currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=1.0, fill=False, color=str(z),alpha=alph))
  # Draw max scoring box
  b = boxes[s[-1]][1:]
  currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=2.0, fill=False, color=(1,0,0)))
  currentAxis.annotate("Best scoring: Score ={:2.2f}".format(scores[s[-1]]), xy=(b[0],X-b[3]),color='red')
  # Draw best fitting box
  for gt in groundTruth:
    a = map(float,gt)
    overlaps = [IoU(a,box[1:]) for box in boxes ]
    best = np.argmax( np.array(overlaps) )
    b = boxes[best][1:]
    currentAxis.add_patch(Rectangle((a[0], X-a[1]), a[2]-a[0], a[1]-a[3], linewidth=0.0, fill=True, color=(0,1,0),alpha=0.3))
    currentAxis.add_patch(Rectangle((b[0], X-b[1]), b[2]-b[0], b[1]-b[3], linewidth=0.0, fill=True, color=(0,0,1),alpha=0.3))
    currentAxis.annotate("Best fitting: Score ={:2.2f} Overlap ={:2.2f}".format(scores[best],overlaps[best]), xy=(b[0],X-b[1]),color='white')

  plt.show()

def histogramOfAreas(boxesFile):
  import matplotlib.pyplot as plt
  boxes = [x.split() for x in open(boxesFile)]
  areas = []
  areasPerImage = {}
  for box in boxes:
    a = area( map(int,box[1:]) )
    try:
      areasPerImage[ box[0] ].append(a)
    except:
      areasPerImage[ box[0] ] = [a]
    areas.append(a)
  large,medium = 200*200,80*80

  largeAreas,mediumAreas,smallAreas = [],[],[]
  for a in areas:
    if a >= large:
      largeAreas.append(a)
    elif a < large and a >= medium:
      mediumAreas.append(a)
    else:
      smallAreas.append(a)
  print 'Small:',len(smallAreas),'( {:.2f}'.format(100*len(smallAreas)/float(len(areas))),'% )'
  print 'Medium:',len(mediumAreas),'( {:.2f}'.format(100*len(mediumAreas)/float(len(areas))),'% )'
  print 'Big:',len(largeAreas),'( {:.2f}'.format(100*len(largeAreas)/float(len(areas))),'% )'

  plt.figure(1)
  ax1 = plt.subplot(1,3,1)
  ax1.set_title('Small:'+'( {:.2f}'.format(100*len(smallAreas)/float(len(areas)))+'% )')
  plt.hist(smallAreas,bins=100)
  ax2 = plt.subplot(1,3,2)
  ax2.set_title('Medium:'+'( {:.2f}'.format(100*len(mediumAreas)/float(len(areas)))+'% )')
  plt.hist(mediumAreas,bins=100)
  ax3 = plt.subplot(1,3,3)
  ax3.set_title('Big:'+'( {:.2f}'.format(100*len(largeAreas)/float(len(areas)))+'% )')
  plt.hist(largeAreas,bins=100)
  plt.show()

  largeAreas,mediumAreas,smallAreas = [],[],[]
  for img in areasPerImage.keys():
    l,m,s=0,0,0
    for a in areasPerImage[img]:
      if a >= large:
        l += 1
      elif a < large and a >= medium:
        m += 1
      else:
        s += 1
    t = len(areasPerImage[img])/100.
    largeAreas.append( l/t )
    mediumAreas.append( m/t )
    smallAreas.append( s/t )

  l = str(int(np.sqrt(large)))
  m = str(int(np.sqrt(medium)))
  plt.figure(1)
  ax1 = plt.subplot(1,3,1)
  ax1.set_title('Small < '+m+'x'+m)
  plt.hist(smallAreas,bins=100)
  ax2 = plt.subplot(1,3,2)
  ax2.set_title('Medium >= '+m+'x'+m+' < '+l+'x'+l)
  plt.hist(mediumAreas,bins=100)
  ax3 = plt.subplot(1,3,3)
  ax3.set_title('Big >= '+l+'x'+l)
  plt.hist(largeAreas,bins=100)
  plt.show()

