# https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet
import os, sys
import Image
from utils import tic, toc
import numpy as np
from skimage import io
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
    data[j,:,:,:] = np.asarray(bIm.resize((227,227), Image.BILINEAR))[:, :, ::-1] # Resize and convert to BGR # BICUBIC
    j += 1
  data -= imagenet.IMAGENET_MEAN[14:241,14:241,:]
  return data.swapaxes(2, 3).swapaxes(1, 2)

def parallelPrepareImg(img, info, name, idx):
  # Make Color Image
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  # Prepare processes
  numProcs = 3
  taskQueue = JoinableQueue()
  resultQueue = ProcQueue()
  processes = []
  for i in range(numProcs):
    t = Process(target=singleWindowProcess, args=(taskQueue, resultQueue, img))
    t.daemon = True
    t.start()
    processes.append(t)
  j = 0
  # Add tasks to the queue
  for b in info:
    idx.write(b[4])
    taskQueue.put( (b,j) )
    j += 1
  for i in range(len(processes)):
    taskQueue.put('stop')
  # Collect results
  data = np.zeros([len(info), 227, 227, 3])
  retrieved = 0
  while retrieved < len(info):
    j,win = resultQueue.get()
    data[j,:,:,:] = win
    retrieved += 1
  # Substract mean and return
  data -= imagenet.IMAGENET_MEAN[14:241,14:241,:]
  return data.swapaxes(2, 3).swapaxes(1, 2)

def singleWindowProcess(inQueue, outQueue, img):
  for data in iter(inQueue.get,'stop'):
    b,j = data
    window = getWindow(img,b)
    bIm = Image.fromarray(np.uint8(window))
    #qw = np.asarray(bIm.resize((227,227), Image.BILINEAR))
    #print qw[0,0,0],qw[0,0,1],qw[0,0,2],qw[0,1,0],qw[0,1,1],qw[0,1,2]
    outQueue.put( (j,np.asarray(bIm.resize((227,227), Image.BICUBIC))[:, :, ::-1]) ) 
    inQueue.task_done()
  inQueue.task_done()
  return True

def emptyMatrix(size):
  data = np.zeros(size)
  return data.astype(np.float32)

def saveMatrix(matrix,outFile):
  outf = open(outFile,'w')
  np.savez_compressed(outf,matrix)
  outf.close()

def computeFeatures(batch, n, data, net, layers, output):
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
    #print outputBlobs[0][0].shape, np.argmax(outputBlobs[0][0]), np.max(outputBlobs[0][0])
    for l in layers.keys():
      allFeat[l][start:finish,:] = outputs[layers[l]['idx']].data[0:elems,:,:,:].reshape([elems,layers[l]['dim']])
  # Save files for this image
  for l in layers.keys():
    saveMatrix(allFeat[l][0:n,:],output+'.'+l)
  lap = toc('Image ready with '+str(n)+' boxes in:',startTime)

def worker(inQueue, net, layers, batch):
  for taskParams in iter(inQueue.get,'stop'):
    n, data, output = taskParams
    computeFeatures(batch, n, data, net, layers, output)
    inQueue.task_done()
  inQueue.task_done()
  return True

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
layers = {'fc6_neuron_cudanet_out': {'dim':4096,'idx':15}, 'conv3_cudanet_out': {'dim':64896,'idx':9}}

batch = 200
taskQueue = Queue.Queue()
p = threading.Thread(target=worker, args=(taskQueue, net, layers, batch))
p.daemon = True
p.start()

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
  img = io.imread(imgsDir+'/'+name+'.jpg')
  indexFile = open(outDir+'/'+name+'.idx','w')
  boxes = parallelPrepareImg(img, images[name], name, indexFile)
  indexFile.close()
  lap = toc('Preparing '+str(len(boxes))+' boxes '+name,lap)
  # Prepare data structures
  taskQueue.put( (boxes.shape[0], boxes, outDir+'/'+name) )
  while taskQueue.qsize() >= 3:
    time.sleep(2)

taskQueue.put('stop')
taskQueue.join()
toc('Total processing time:',startTime)

