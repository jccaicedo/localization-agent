import os,sys
import utils as cu
import scipy.io
import h5py
import numpy as np
import libDetection as det

params = cu.loadParams('rcnnImdbFile MatlabScoresDir outputDir doNMS')

imdb = h5py.File(params['rcnnImdbFile'])
print 'Images:',imdb['imdb']['image_ids']
images = [u''.join(unichr(c) for c in imdb[o]) for o in imdb['imdb']['image_ids'][0]]
doNMS = params['doNMS'] != 'noNMS'

for f in os.listdir(params['MatlabScoresDir']):
  if f.endswith('.mat') and f.find('_boxes_') != -1:
    nameParts = f.split('_')
    out = open(params['outputDir'] + '/' + nameParts[0] + '_' + nameParts[1] + '_det.out', 'w')
    data = scipy.io.loadmat(params['MatlabScoresDir'] + '/' + f)
    print nameParts[0:2]
    for i in range(data['boxes'].shape[0]):
      detections = data['boxes'][i][0]
      img = str(images[i])
      boxes = [ box[0:4].tolist() for box in detections ]
      scores = [ box[-1] for box in detections ]
      if len(boxes) == 0: 
        continue
      if doNMS:
        boxes, scores = det.nonMaximumSuppression(boxes, scores, 0.3)
      for j in range(len(boxes)):
        box = boxes[j]
        out.write(img + ' {:10.8f} {:.0f} {:.0f} {:.0f} {:.0f} 0\n'.format(scores[j], box[0], box[1], box[2], box[3]) )
     
