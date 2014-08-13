# https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet
import os, sys
import Image
from utils import tic, toc
import numpy as np
#from skimage import io
import threading
import Queue
from multiprocessing import Process, JoinableQueue
from multiprocessing import Queue as ProcQueue
import time

##################################
# Parameter checking
#################################
if len(sys.argv) < 3:
  print 'Use: extractCNNFeatures.py bboxes imgsDir outputDir'
  sys.exit()

images = [x.replace('\n','') for x in open(sys.argv[1])]
imgsDir = sys.argv[2]
outDir = sys.argv[3]

from caffe import imagenet
#MODEL_FILE = '/u/sciteam/caicedor/models/imagenet_deploy.prototxt'
MODEL_FILE = '/home/caicedo/workspace/caffe/examples/imagenet/alexnet_deploy.prototxt.200' 
#PRETRAINED = '/u/sciteam/caicedor/models/finetune_voc_2012_train_iter_70k'
#PRETRAINED = '/home/caicedo/software/rcnn-master/data/caffe_nets/finetune_voc_2012_train_iter_70k'
PRETRAINED = '/home/caicedo/software/rcnn-master/data/caffe_nets/ilsvrc_2012_train_iter_310k.original'

IMG_DIM = 256
CROP_SIZE = 227
CONTEXT_PAD = 0
#meanImage = '/u/sciteam/caicedor/scratch/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
meanImage = '/home/caicedo/workspace/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
net = imagenet.ImageNetClassifier(MODEL_FILE, PRETRAINED, IMAGE_DIM=IMG_DIM, CROPPED_DIM=CROP_SIZE, MEAN_IMAGE=meanImage)
net.caffenet.set_phase_test()
net.caffenet.set_mode_gpu()

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

def generateBoxes(height, width, stride, scale, cropSize, limitX, limitY):
  if height < cropSize:
    sf = float(cropSize)/height
    width = int(width*sf)
    stride = int(stride*sf)
    scale = scale*sf
    height = cropSize
  if width < cropSize:
    sf = float(cropSize)/width
    height = int(height*sf)
    stride = int(stride*sf)
    scale = scale*sf
    width = cropSize
  X1 = [stride*i for i in range((width-cropSize)/stride)]
  Y1 = [stride*i for i in range((height-cropSize)/stride)]
  if len(X1) == 0: X1 = [0]
  if len(Y1) == 0: Y1 = [0]
  strideCorrectionX = (width - (X1[-1] + cropSize))/ max(1,len(X1)-1)
  strideCorrectionY = (height - (Y1[-1] + cropSize))/ max(1,len(Y1)-1)
  X1 = [X1[i] + i*strideCorrectionX for i in range(len(X1))]
  Y1 = [Y1[i] + i*strideCorrectionY for i in range(len(Y1))]
  boxes = []
  for i in X1:
    for j in Y1:
      boxes.append( [int(i/scale),int(j/scale),
                     min(int((i+cropSize)/scale),limitX-1), min(int((j+cropSize)/scale), limitY-1)] )
  return boxes

def multiScaleBoxes(dims, cropSize):
  scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
  stride = 32
  boxes = []
  for i in range(len(scales)):
    bAtScale = generateBoxes(int(dims[0]*scales[i]), int(dims[1]*scales[i]), int(stride*scales[i]), scales[i], cropSize, dims[1], dims[0])
    boxes += bAtScale
  return boxes

def processImg(imgName, filename, idx, batchSize, layers, output):
  startTime = tic()
  # Initialize image and boxes
  dims = net.caffenet.InitializeImage(filename, IMG_DIM, ImageNetMean, CROP_SIZE)
  boxes = multiScaleBoxes(dims, CROP_SIZE)
  # Write index file
  [idx.write(imgName + ' ' + ' '.join(map(str,b)) + '\n') for b in boxes]
  #Prepare boxes, make sure that extra rows are added to fill the last batch
  allFeat = {}
  n = len(boxes)
  for l in layers.keys():
    allFeat[l] = emptyMatrix([n,layers[l]['dim']])
  numBatches = (n + batchSize - 1) / batchSize
  boxes += [ [0,0,0,0] for x in range(numBatches * batchSize - n) ]

  for k in range(numBatches):
    s,f = k*batchSize,(k+1)*batchSize
    e = batchSize if f <= n else n-s
    # Forward this batch
    net.caffenet.ForwardRegions(boxes[s:f],CONTEXT_PAD) #,filename)
    outputs =  net.caffenet.blobs
    f = n if f > n else f
    # Collect outputs
    for l in layers.keys():
      allFeat[l][s:f,:] = outputs[layers[l]['idx']].data[0:e,:,:,:].reshape([e,layers[l]['dim']])
  # Release image data
  net.caffenet.ReleaseImageData()
  # Save files for this image
  for l in layers.keys():
    saveMatrix(allFeat[l][0:n,:],output+'.'+l)
  lap = toc('GPU is done with '+str(n)+' boxes in:',startTime)

def emptyMatrix(size):
  data = np.zeros(size)
  return data.astype(np.float32)

def saveMatrix(matrix,outFile):
  outf = open(outFile,'w')
  np.savez_compressed(outf,matrix)
  outf.close()

#################################
# Extract Features
#################################
startTime = tic()
totalItems = len(images)
layers = {'fc6_neuron_cudanet_out': {'dim':4096,'idx':'fc6'}, 'fc7_neuron_cudanet_out': {'dim':4096,'idx':'fc7'}}
batch = 50

print 'Extracting features for',totalItems,'total images'
for name in images:
  # Check if files already exist
  processed = 0
  for l in layers.keys():
    if os.path.isfile(outDir+'/'+name+'.'+l):
      processed += 1
  if processed == len(layers):
    continue
  # Get features for patches
  indexFile = open(outDir+'/'+name+'.idx','w')
  processImg(name, imgsDir+'/'+name+'.jpg', indexFile, batch, layers, outDir+'/'+name)
  indexFile.close()

toc('Total processing time:',startTime)

