import os,sys
import utils as cu
import libDetection as det
import numpy as np

params = cu.loadParams('scoresFile outputDir category')

imgs = {}
data = open(params['scoresFile'])
l = data.readline()
while l != '':
  d = l.split()
  rec = {'b': d[0:5], 's':map(float,d[5:])}
  try: 
    imgs[d[0]]['boxes'].append(rec['b'])
    imgs[d[0]]['scores'] = np.vstack( (imgs[d[0]]['scores'], np.array(rec['s'])) )
  except: 
    imgs[d[0]] = {'boxes':[], 'scores':[]}
    imgs[d[0]]['boxes'] = [ rec['b'] ]
    imgs[d[0]]['scores'] = np.array(rec['s'])
  l = data.readline()
data.close()

categories = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
categories = categories + [c + '_big' for c in categories] + [c + '_inside' for c in categories]

if params['category'] == 'all':
  selectedCategories = range(len(categories))
  out = dict([ (c,open(params['outputDir']+'/'+c+'.out','w')) for c in categories])
else:
  catIdx = int(params['category'])
  selectedCategories = [catIdx]
  out = {categories[catIdx]:open(params['outputDir']+'/'+categories[catIdx]+'.out','w')}

for i in imgs.keys():
  print i,
  for j in selectedCategories:
    print categories[j],
    fb, fs = det.nonMaximumSuppression(imgs[i]['boxes'], imgs[i]['scores'][:,j], 0.3)
    for k in range(len(fb)):
      out[categories[j]].write(i + ' {:.8f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(fs[k],fb[k][0],fb[k][1],fb[k][2],fb[k][3],0))
  print ''
  imgs[i] = []

for o in out.keys():
  out[o].close()
