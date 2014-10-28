# https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet
import os, sys
from utils import tic, toc
import numpy as np
import scipy.io

##################################
# Parameter checking
#################################
if len(sys.argv) < 5:
  print 'Use: extractFeaturesForRCNN.py bboxes imgsDir groundTruthFile outputDir'
  sys.exit()

bboxes  = [ (x,x.split()) for x in open(sys.argv[1])]
imgsDir = sys.argv[2]
groundTruthFile = sys.argv[3]
outDir = sys.argv[4]

from caffe import wrapperv0

MODEL_FILE = '/home/caicedo/workspace/rcnn/model-defs/rcnn_batch_256_output_fc7.old_format.prototxt'
PRETRAINED = '/home/caicedo/workspace/rcnn/data/caffe_nets/finetune_voc_2012_train_iter_70k'
IMG_DIM = 256
CROP_SIZE = 227
CONTEXT_PAD = 0
batch = 50

meanImage = '/home/caicedo/workspace/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
net = wrapperv0.ImageNetClassifier(MODEL_FILE, PRETRAINED, IMAGE_DIM=IMG_DIM, CROPPED_DIM=CROP_SIZE, MEAN_IMAGE=meanImage)
net.caffenet.set_mode_gpu()
net.caffenet.set_phase_test()

ImageNetMean = net._IMAGENET_MEAN.swapaxes(1, 2).swapaxes(0, 1).astype('float32')

##################################
# Functions
#################################

def getWindow(img, box):
  dx = int( float(box[2]-box[0])*0.10 )
  dy = int( float(box[3]-box[1])*0.10 )
  x1 = max(box[0]-dx,0)
  x2 = min(box[2]+dx,img.shape[1])
  y1 = max(box[1]-dy,0)
  y2 = min(box[3]+dy,img.shape[0])
  return img[ y1:y2, x1:x2, : ]

def processImg(info, filename, batchSize, layers):
  startTime = tic()
  allFeat = {}
  n = len(info)
  for l in layers.keys():
    allFeat[l] = emptyMatrix([n,layers[l]['dim']])
  numBatches = (n + batchSize - 1) / batchSize
  # Prepare boxes, make sure that extra rows are added to fill the last batch
  boxes = [x[:-1] for x in info] + [ [0,0,0,0] for x in range(numBatches * batchSize - n) ]
  # Initialize the image
  net.caffenet.InitializeImage(filename, IMG_DIM, ImageNetMean, CROP_SIZE)
  for k in range(numBatches):
    s,f = k*batchSize,(k+1)*batchSize
    e = batchSize if f <= n else n-s
    # Forward this batch
    net.caffenet.ForwardRegions(boxes[s:f],CONTEXT_PAD) #,filename)
    outputs = net.caffenet.blobs
    f = n if f > n else f
    # Collect outputs
    for l in layers.keys():
      allFeat[l][s:f,:] = outputs[layers[l]['idx']].data[0:e,:,:,:].reshape([e,layers[l]['dim']])
  # Release image data
  net.caffenet.ReleaseImageData()
  # Return features of boxes for this image
  for l in layers.keys():
    allFeat[l] = allFeat[l][0:n,:]
  lap = toc('GPU is done with '+str(len(info))+' boxes in:',startTime)
  return allFeat

def emptyMatrix(size):
  data = np.zeros(size)
  return data.astype(np.float32)

def saveMatrix(matrix,outFile):
  outf = open(outFile,'w')
  np.savez_compressed(outf,matrix)
  outf.close()

def loadBoxAnnotationsFile(filename):
  categories = set()
  gt = [x.split() for x in open(filename)]
  images = {}
  for k in gt:
    categories.add(k[1])
    try:
      images[k[0]] += [ [k[1]] + map(float,k[2:]) ]
    except:
      images[k[0]] = [ [k[1]] + map(float,k[2:]) ]
  categories = list(categories)
  categories.sort()
  catIdx = dict([(categories[i],i) for i in range(len(categories))])
  return images, catIdx

def createAndSaveMatlabFile(boxes, feat, groundTruths, categories, output):
  mat = {}
  mat['__header__'] = 'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sun Oct 19 19:23:06 2014';
  mat['__globals__'] = []
  mat['__version__'] = '1.0';
  mat['boxes'] = np.array(boxes)
  mat['feat'] = feat
  mat['overlap'] = np.zeros( (feat.shape[0], len(categories)) )
  mat['class'] = np.zeros( (feat.shape[0], 1) )
  mat['gt'] = np.zeros( (feat.shape[0], 1) )
  duplicate = []
  for gt in groundTruths:
    for i in range(mat['boxes'].shape[0]):
      box = mat['boxes'][i,:].tolist()
      iou = det.IoU(box, gt[1:])
      if iou > mat['overlap'][i, categories[gt[0]]]:
        mat['overlap'][i, categories[gt[0]]] = iou
      if iou == 1:
        mat['gt'][i] = 1.0
        if mat['class'][i] == 0 or mat['class'][i] == categories[gt[0]]+1:
          mat['class'][i] = categories[gt[0]]+1
        else:
          duplicate.append( {'row':i, 'class':categories[gt[0]]+1} )
  shift = 0
  for d in duplicate:
    for key in ['feat','gt','boxes','overlap','class']:
      mat[key] = np.vstack( (mat[key][d['row'] + shift,:], mat[key]) )
    shift += 1
    mat['class'][0] = d['class']
  scipy.io.savemat(output, mat, do_compression=True)

##################################
# Organize boxes by source image
#################################
startTime = tic()

images = {}
for s,box in bboxes:
  # Subtract 1 because RCNN proposals have 1-based indexes for Matlab
  b = map(lambda x: int(x)-1,box[1:]) + [s.replace('\n','')]
  try:
    images[ box[0] ].append(b)
  except:
    images[ box[0] ] = [b]
lap = toc('Reading boxes file:',startTime)

groundTruth, categories = loadBoxAnnotationsFile(groundTruthFile)
print 'Found categories:',categories
lap = toc('Reading ground truth file:',lap)

#################################
# Extract Features
#################################
totalItems = len(bboxes)
del(bboxes)
layers = {'pool5': {'dim':9216,'idx':'pool5'}}

print 'Extracting features for',totalItems,'total regions'
for name in images.keys():
  # Get window proposals
  boxes = images[name]
  try: gt = groundTruths[name]
  except: gt = []
  [boxes.add(b) for b in gt]
  feat = processImg(boxes, imgsDir+'/'+name+'.jpg', batch, layers)
  print 'pool5',feat['pool5'].shape, np.sum(np.sum(feat['pool5']))
  createAndSaveMatlabFile(boxes, feat['pool5'], gt, categories, outDir+'/'+name+'.mat')

toc('Total processing time:',startTime)

