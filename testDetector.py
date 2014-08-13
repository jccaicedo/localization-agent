import os,sys
import numpy as np
import utils as cu
import libLearning as learn
import libDetection as det
from sklearn import svm

## Main Program Parameters
if len(sys.argv) < 8:
  print "Use: testDetector.py modelFile testImage featuresDir featuresExt maxOverlap threshold groundTruth"
  sys.exit()

#model = cu.loadModel(sys.argv[1])
model = det.loadSVMDetector('linear',sys.argv[1])
testImage = sys.argv[2]
featuresDir = sys.argv[3]
featuresExt = sys.argv[4]
maxOverlap = float(sys.argv[5])
threshold = float(sys.argv[6])
gt = [x.split() for x in open(sys.argv[7])]
groundTruth = {}
for k in gt:
  try:
    groundTruth[k[0]] += [ k[1:] ]
  except:
    groundTruth[k[0]] = [ k[1:] ]

## Make Detections
features,bboxes = cu.loadMatrixAndIndex( featuresDir+'/'+testImage+'.'+featuresExt )
scores = model.predict( features )
candIdx = scores>=threshold
numCandidates = candIdx[candIdx==True].shape[0]
print 'Candidate Boxes:',numCandidates
if numCandidates > 0:
  candidateBoxes = [bboxes[t] for t in range(candIdx.shape[0]) if candIdx[t]]
  candidateScores = scores[candIdx]
  filteredBoxes,filteredScores = det.nonMaximumSuppression(candidateBoxes,candidateScores,maxOverlap)
  print testImage,len(filteredBoxes)
  #for i in range(len(filteredBoxes)):
  #  b = filteredBoxes[i]
  #  print b[0] + ' {:.8f} {:} {:} {:} {:}\n'.format(filteredScores[i],b[1],b[2],b[3],b[4])
  det.showDetections('/home/caicedo/data/allimgs/'+testImage+'.jpg', filteredBoxes, filteredScores, True)
  det.showDetections('/home/caicedo/data/allimgs/'+testImage+'.jpg', candidateBoxes, candidateScores, False)
  det.showBestMatches('/home/caicedo/data/allimgs/'+testImage+'.jpg', candidateBoxes, candidateScores, groundTruth[testImage])

sys.exit()
import matplotlib.pyplot as plt
features = np.asmatrix(features)
K = features*features.T
N = np.diag(K)
D = np.tile(np.mat(N).T,(1,K.shape[0])) + np.tile(np.mat(N),(K.shape[0],1)) - 2*K
plt.imshow(G)
plt.colorbar()
plt.show()

boxes = [ map(float,x[1:]) for x in bboxes]
O = np.zeros(G.shape)
for i in range(O.shape[0]):
  for j in range(O.shape[1]):
    O[i,j] = det.IoU(boxes[i],boxes[j])

plt.imshow(O)
plt.colorbar()
plt.show()
