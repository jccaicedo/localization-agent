# https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet
import os, sys
from utils import tic, toc
import numpy as np
import scipy.io

##################################
# Parameter checking
#################################
if len(sys.argv) < 5:
  print 'Use: extractCNNFeatures.py bboxes imgsDir rcnnModel outputFile'
  sys.exit()

bboxes  = [ (x,x.split()) for x in open(sys.argv[1])]
imgsDir = sys.argv[2]
rcnnModel = scipy.io.loadmat(sys.argv[3])
outFile = sys.argv[4]

W = rcnnModel['integratedModel'][0,0]['W']
B = rcnnModel['integratedModel'][0,0]['B']
C = [str(c[0][0]) for c in rcnnModel['integratedModel'][0,0]['classes'].tolist()]
del rcnnModel

from caffe import imagenet
MODEL_FILE = '/home/caicedo/workspace/rcnn/model-defs/rcnn_batch_256_output_fc7.old_format.prototxt'
PRETRAINED = '/home/caicedo/workspace/rcnn/data/caffe_nets/finetune_voc_2012_train_iter_70k'
IMG_DIM = 256
CROP_SIZE = 227
CONTEXT_PAD = 0
batch = 50

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

#################################
# Extract Features
#################################
totalItems = len(bboxes)
del(bboxes)
layers = {'fc7': {'dim':4096,'idx':'fc7'}}

print 'Extracting features for',totalItems,'total images'
indexFile = open(outFile,'w')
for name in images.keys():
  # Check if files already exist
  processed = 0
  # Get window proposals
  feat = processImg(images[name], imgsDir+'/'+name+'.jpg', batch, layers)
  scores = np.dot(feat['fc7'], W) + B
  # Write the index file
  for i in range(len(images[name])):
    indexFile.write(images[name][i][4] + ' ')
    indexFile.write( ' '.join( map(str, scores[i,:].tolist() )) + '\n' )
  
indexFile.close()

toc('Total processing time:',startTime)

