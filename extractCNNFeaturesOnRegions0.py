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

bboxes  = [ (x,x.split()) for x in open(sys.argv[1])]
imgsDir = sys.argv[2]
outDir = sys.argv[3]

from caffe import imagenet
MODEL_FILE = '/home/caicedo/software/caffe-master/examples/imagenet_deploy.prototxt'
#PRETRAINED = '/home/caicedo/Downloads/caffe_reference_imagenet_model'
PRETRAINED = '/home/caicedo/software/rcnn-master/data/caffe_nets/finetune_voc_2012_train_iter_70k'
net = imagenet.ImageNetClassifier(MODEL_FILE, PRETRAINED)
net.caffenet.set_phase_test()
net.caffenet.set_mode_gpu()
ImageNetMean = imagenet.IMAGENET_MEAN.swapaxes(1, 2).swapaxes(0, 1).astype('float32')

# From: https://github.com/zachrahan/scikits-image/blob/master/skimage/io/tests/test_freeimage.py?source=cc
#try:
#  import skimage.io._plugins.freeimage_plugin as fi
#  FI_available = True
#  io.use_plugin('freeimage')
#except RuntimeError:
#  FI_available = False

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

def processImg(info, filename, idx, batchSize, layers, output):
  startTime = tic()
  allFeat = {}
  n = len(info)
  for l in layers.keys():
    allFeat[l] = emptyMatrix([n,layers[l]['dim']])
  numBatches = (n + batchSize - 1) / batchSize
  # Write the index file
  [idx.write(b[4]) for b in info]
  # Prepare boxes, make sure that extra rows are added to fill the last batch
  boxes = [x[:-1] for x in info] + [ [0,0,0,0] for x in range(numBatches * batchSize - n) ]
  # Initialize the image
  net.caffenet.InitializeImage(filename, ImageNetMean)
  for k in range(numBatches):
    s,f = k*batchSize,(k+1)*batchSize
    e = batchSize if f <= n else n-s
    # Forward this batch
    net.caffenet.ForwardRegions(boxes[s:f])
    #outputBlobs = [ np.empty((batch, 1000, 1, 1), dtype=np.float32) ]
    #net.caffenet.ForwardRegions(boxes[s:f],filename, outputBlobs)
    #print outputBlobs[0][0].shape, np.argmax(outputBlobs[0][0]), np.max(outputBlobs[0][0])
    outputs =  net.caffenet.blobs()
    f = n if f > n else f
    # Collect outputs
    for l in layers.keys():
      allFeat[l][s:f,:] = outputs[layers[l]['idx']].data[0:e,:,:,:].reshape([e,layers[l]['dim']])
  # Release image data
  net.caffenet.ReleaseImageData()
  # Save files for this image
  for l in layers.keys():
    saveMatrix(allFeat[l][0:n,:],output+'.'+l)
  lap = toc('GPU is done with '+str(len(info))+' boxes in:',startTime)

def emptyMatrix(size):
  data = np.zeros(size)
  return data.astype(np.float32)

def saveMatrix(matrix,outFile):
  outf = open(outFile,'w')
  np.savez_compressed(outf,matrix)
  outf.close()

##################################
# Organize boxes by source image
#################################
startTime = tic()

images = {}
for s,box in bboxes:
  b = map(int,box[1:]) + [s]
  try:
    images[ box[0] ].append(b)
  except:
    images[ box[0] ] = [b]
lap = toc('Reading boxes file:',startTime)

#################################
# Extract Features
#################################
totalItems = len(bboxes)
del(bboxes)
layers = {'fc6_neuron_cudanet_out': {'dim':4096,'idx':15}}
#layers = {'fc6_neuron_cudanet_out': {'dim':4096,'idx':15},'conv3_cudanet_out': {'dim':64896,'idx':9}}

batch = 200

print 'Extracting features for',totalItems,'total images'
for name in images.keys():
  # Check if files already exist
  processed = 0
  for l in layers.keys():
    if os.path.isfile(outDir+'/'+name+'.'+l):
      processed += 1
  if processed == len(layers):
    continue
  # Get window proposals
  indexFile = open(outDir+'/'+name+'.idx','w')
  processImg(images[name], imgsDir+'/'+name+'.jpg', indexFile, batch, layers, outDir+'/'+name)
  indexFile.close()

toc('Total processing time:',startTime)

