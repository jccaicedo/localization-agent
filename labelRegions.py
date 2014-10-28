import os,sys
import utils as cu
import libDetection as ldet
import numpy as np

params = cu.loadParams('scoresFile groundTruth relation output')
scores = [x.split() for x in open(params['scoresFile'])]
ground = cu.loadBoxIndexFile(params['groundTruth'])
scores.sort(key=lambda x:float(x[1]), reverse=True)

if params['relation'] == 'big': 
  operator = lambda x,y: np.exp( -( (1.0-ldet.overlap(x,y))**2 + (0.25-ldet.IoU(x,y))**2 ) ) >= 0.7
if params['relation'] == 'inside': 
  operator = lambda x,y: np.exp( -( (1.0-ldet.overlap(y,x))**2 + (0.25-ldet.IoU(x,y))**2 ) ) >= 0.7
if params['relation'] == 'tight': 
  operator = lambda x,y: ldet.IoU(x,y) >= 0.5

out = open(params['output'],'w')
for s in scores:
  box = map(float,s[2:7])
  img = s[0]
  try: gtBoxes = ground[img]
  except: gtBoxes = []
  match = '0'
  for gt in gtBoxes:
    if operator(box,gt):
      match = '1'
  out.write(' '.join(s) + ' ' + match + '\n')
out.close()
