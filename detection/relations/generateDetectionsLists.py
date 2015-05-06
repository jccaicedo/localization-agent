import os,sys
import utils as cu
import libDetection as det
import numpy as np
import pickle

params = cu.loadParams('scoresFile outputDir category')

if os.path.isfile(params['scoresFile']+'.p'):
  print 'Loading pickled data'
  imgs = pickle.load( open(params['scoresFile']+'.p', 'rb') )
else:
  imgs = {}
  data = open(params['scoresFile'])
  l = data.readline()
  counter = 0
  while l != '':
    counter += 1
    d = l.split()
    rec = {'b': map(float,d[1:5]), 's':map(float,d[5:])}
    try: 
      imgs[d[0]]['boxes'].append(rec['b'])
      imgs[d[0]]['scores'] = np.vstack( (imgs[d[0]]['scores'], np.array(rec['s'])) )
    except: 
      imgs[d[0]] = {'boxes':[], 'scores':[]}
      imgs[d[0]]['boxes'] = [ rec['b'] ]
      imgs[d[0]]['scores'] = np.array(rec['s'])
    l = data.readline()
    if counter % 100000 == 0: print counter
  data.close()
  
  pickle.dump(imgs, open(params['scoresFile']+'.p','wb'))

sys.exit()

categories = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
categories = categories + [c + '_big' for c in categories] + [c + '_inside' for c in categories]

if params['category'] == 'all':
  selectedCategories = range(len(categories))
  out = dict([ (c,open(params['outputDir']+'/'+c+'.out','w')) for c in categories])
else:
  catIdx = int(params['category'])
  selectedCategories = [catIdx]
  out = {categories[catIdx]:open(params['outputDir']+'/'+categories[catIdx]+'.out','w')}

counter = 0
for i in imgs.keys():
  counter += 1
  print counter,i,
  for j in selectedCategories:
    print categories[j],
    fb, fs = det.nonMaximumSuppression(imgs[i]['boxes'], imgs[i]['scores'][:,j], 0.3)
    for k in range(len(fb)):
      out[categories[j]].write(i + ' {:.8f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(fs[k],fb[k][0],fb[k][1],fb[k][2],fb[k][3],0))
  print ''
  imgs[i] = []

for o in out.keys():
  out[o].close()
