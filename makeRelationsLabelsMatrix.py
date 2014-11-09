import os,sys
import utils as cu
import libDetection as det

import cPickle as pickle
import scipy.io
import numpy as np

params = cu.loadParams('dbDir relationsFile outputDir')

archive = {}

T = pickle.load( open(params['dbDir']+'/db.idx','rb') )
M = scipy.io.loadmat(params['dbDir']+'/db.cache')

archive['images'] = T.keys()
index = np.zeros( (len(T), 2), np.int )
for i in range(len(archive['images'])):
  idx = T[archive['images'][i]]
  index[i,0] = idx['s'] + 1
  index[i,1] = idx['e']
archive['index'] = index

data = [x.split() for x in open(params['relationsFile'])]
categories = set()
labels = {}
for d in data:
  r = [d[1]] + map(float,d[2:])
  try: labels[d[0]].append( r )
  except: labels[d[0]] =  [ r ]
  categories.add(d[1])
categories = list(categories)
categories.sort()
C = dict( [ (categories[c],c) for c in range(len(categories))] )

print 'Identifying labeled boxes'
L = np.zeros( (M['B'].shape[0], 60), np.int32 )
for img in labels.keys():
  print img
  idx = T[img]
  for j in range(idx['s'],idx['e']):
    box = M['B'][j,:].tolist()
    for l in labels[img]:
      iou = det.IoU(box,l[1:])
      if iou == 1.0:
        L[j,C[l[0]]] = 1

archive['labels'] = L

scipy.io.savemat( params['outputDir']+'/boxes.mat', {'boxes:':M['B']}, do_compression=True )
scipy.io.savemat( params['outputDir']+'/scores.mat', {'scores:':M['S']} )
scipy.io.savemat( params['outputDir']+'/labels.mat', {'labels:':L} )
scipy.io.savemat( params['outputDir']+'/index.mat', {'index:':index, 'images':T.keys()}, do_compression=True )





