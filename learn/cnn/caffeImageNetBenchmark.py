# https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet
import os, sys
import Image
from utils import tic, toc

##################################
# Parameter checking
#################################
if len(sys.argv) < 3:
  print 'Use: benchmark.py imageList imgsDir outputFile'
  print '     localMachine: /home/caicedo/software/decaf-release-master/pretrained/'
  print '     cluster:      /projects/VisionLanguage/caicedo/data/decaf/'
  sys.exit()

images  = [x.replace('\n','') for x in open(sys.argv[1])]
imgsDir = sys.argv[2]
outFile = sys.argv[3]

import numpy as np
from skimage import io
#from decaf.scripts.imagenet import DecafNet
from decaf.util import transform
#net = DecafNet(pretrainedModel+'/imagenet.decafnet.epoch90', pretrainedModel+'/imagenet.decafnet.meta')

from caffe import imagenet
MODEL_FILE = '/home/caicedo/software/caffe-master/examples/imagenet_deploy.prototxt'
PRETRAINED = '/home/caicedo/Downloads/caffe_reference_imagenet_model'
net = imagenet.ImageNetClassifier(MODEL_FILE, PRETRAINED)
net.caffenet.set_phase_test()
#net.caffenet.set_mode_cpu()
net.caffenet.set_mode_gpu()

# From: https://github.com/zachrahan/scikits-image/blob/master/skimage/io/tests/test_freeimage.py?source=cc
try:
    import skimage.io._plugins.freeimage_plugin as fi
    FI_available = True
    io.use_plugin('freeimage')
except RuntimeError:
    FI_available = False

# Prevent Hitting Memory Limits
r = os.popen('cat /proc/meminfo | grep MemTotal').read()
mem = int(r.split()[1])
if mem < 18282710:
  batch = 100
else:
  batch = 500

##################################
# Functions
#################################

def loadImageNetWords():
  words = [w.split(',')[0] for w in open('/home/caicedo/software/caffe-master/python/caffe/imagenet/ilsvrc_2012_synset_words.txt')]
  words = [' '.join(w.split()[1:]) for w in words]
  return words

#################################
# Classify Images
#################################
totalItems = len(images)
words = loadImageNetWords()
print 'Classifying',totalItems,'total images'
result = open(outFile,'w')
for name in images:
  #img = io.imread(imgsDir+'/'+name)
  # Extract and store CNN Features
  print name
  #scores = net.classify(img, center_only=False)
  prediction = net.predict(imgsDir+'/'+name)
  top1 = words[ np.argmax(prediction) ]
  result.write(name + ' ' + top1 + '\n')

