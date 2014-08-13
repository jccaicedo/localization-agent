# https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet
import os, sys
import timeit
import Image

import numpy as np
from skimage import io

from caffe import imagenet
MODEL_FILE = '/home/caicedo/software/caffe-master/examples/imagenet_deploy.prototxt'
#PRETRAINED = '/home/caicedo/Downloads/caffe_reference_imagenet_model'
PRETRAINED = '/home/caicedo/software/rcnn-master/data/caffe_nets/finetune_voc_2012_train_iter_70k'
net = imagenet.ImageNetClassifier(MODEL_FILE, PRETRAINED)
net.caffenet.set_phase_test()
net.caffenet.set_mode_gpu()

# From: https://github.com/zachrahan/scikits-image/blob/master/skimage/io/tests/test_freeimage.py?source=cc
try:
  import skimage.io._plugins.freeimage_plugin as fi
  FI_available = True
  io.use_plugin('freeimage')
except RuntimeError:
  FI_available = False

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

def prepareImg(img, info, name, idx):
  data = np.zeros([len(info), 227, 227, 3])
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  j = 0
  for b in info:
    idx.write(b[4])
    window = getWindow(img,b)
    bIm = Image.fromarray(np.uint8(window))
    data[j,:,:,:] = np.asarray(bIm.resize((227,227), Image.BICUBIC))[:, :, ::-1] # Resize and convert to BGR
    j += 1
  data -= imagenet.IMAGENET_MEAN[14:241,14:241,:]
  return data.swapaxes(2, 3).swapaxes(1, 2)

def tic():
  return timeit.default_timer()

def toc(msg, prev):
  curr = timeit.default_timer()
  print msg,"%.2f"%(curr-prev),'s'
  return tic()

def emptyMatrix(size):
  data = np.zeros(size)
  return data.astype(np.float32)

def saveMatrix(matrix,outFile,num):
  outf = open(outFile.replace('.',str(num)+'.'),'w')
  np.savez_compressed(outf,matrix)
  outf.close()
  print 'Matrix',matrix.shape,'saved'

def computeFeatures(batch, n, data, net, layers):
  startTime = tic()
  allFeat = {}
  for l in layers.keys():
    allFeat[l] = emptyMatrix([n,layers[l]['dim']])
  # Extract and store CNN Features
  outputBlobs = [np.empty((batch, 1000, 1, 1), dtype=np.float32)]
  for i in range(batch,n+batch,batch):
    inputBlobs = np.empty((batch, 3, 227, 227), dtype=np.float32)
    start = i-batch
    finish = min(i,n)
    elems = finish-start
    inputBlobs[0:elems,:,:,:] = data[start:finish,:,:,:]
    net.caffenet.Forward([inputBlobs], outputBlobs)
    outputs =  net.caffenet.blobs()
    for l in layers.keys():
      allFeat[l][start:finish,:] = outputs[layers[l]['idx']].data[0:elems,:,:,:].reshape([elems,layers[l]['dim']])
  return allFeat

##################################
# Main Program
##################################
if len(sys.argv) < 4:
  print 'Use: extractCNNFeatures.py bboxes imgsDir output'
  sys.exit()

startTime = tic()
bboxes  = [(x,x.split()) for x in open(sys.argv[1])]
imgsDir = sys.argv[2]
outFile = sys.argv[3]

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
indexFile = open(outFile+'.idx','w')
totalItems = len(bboxes)
layers = {'fc6_neuron_cudanet_out': {'dim':4096,'idx':15} }
allFeat = {}
print 'Extracting features for',totalItems,'total images'
for l in layers.keys():
  allFeat[l] = emptyMatrix([totalItems,layers[l]['dim']])
batchSize = 2000
fileRecords = 50000
data = emptyMatrix([batchSize*2, 3, 227, 227])
n,m,p,q,r = 0,0,0,0,0
for name in images.keys():
  img = io.imread(imgsDir+'/'+name+'.jpg')
  boxes = prepareImg(img, images[name], name, indexFile)
  m = n + boxes.shape[0] 
  data[n:m,:,:,:] = boxes
  n = m
  if n > batchSize:
    tmpFeat = computeFeatures(200, n, data, net, layers)
    for l in layers.keys():
      allFeat[l][p:p+n,:] = tmpFeat[l]
    data = emptyMatrix([batchSize*2, 3, 227, 227])
    p = p + n
    r = r + n
    lap = toc('Batch with '+str(n)+' ('+ "%.2f"%(100*(float(r)/float(totalItems)))+'%):',lap)
    n = 0
    if p > fileRecords:
      for l in layers.keys():
        saveMatrix(allFeat[l][0:p,:],outFile+'.'+l,q)
        allFeat[l] = emptyMatrix([n,layers[l]])
      q = q + 1
      p = 0

indexFile.close()
if n < batchSize and n > 0:
  tmpFeat = computeFeatures(200, n, data, net, layers)
  for l in layers.keys():
    allFeat[l][p:p+n,:] = tmpFeat[l] 
  p = p + n
  lap = toc('Final batch '+str(n)+' ('+ "%.2f"%(100*(float(r+n)/float(totalItems)))+'%):',lap)

for l in layers.keys():
  saveMatrix(allFeat[l][0:p,:],outFile+'.'+l,q)

toc('Total processing time:',startTime)

