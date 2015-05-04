import os,sys
import utils as cu
import libDetection as det

def findRelation(box, gt, iou):
  if iou >= 0.6:
    return 'tight'
  else:
    ov = det.overlap(gt,box)
    if ov >= 1.0 and iou <= 0.3 and iou >= 0.2:
      return 'inside'
    ov = det.overlap(box,gt)
    if ov >= 0.9 and iou <= 0.4 and iou >= 0.2:
      return 'big'
  return None
  
def loadRegionsFile(filename):
  regions = {}
  headers = {}
  data = [x.split() for x in open(filename)]
  im = ''
  for i in range(len(data)):
    d = data[i]
    if d[0] == "#":
      im = data[i+1][0].split('/')[-1].replace('.jpg','')
      regions[im] = []
      headers[im] = data[i+1:i+5]
      i += 5
      continue
    elif len(d) == 6:
      regions[im].append(map(float,d[2:]))
  return regions,headers

def loadAllGroundTruths(dir):
  groundTruths = {}
  for f in os.listdir(dir):
    if os.path.isfile(dir + '/' + f):
      category = f.split('_')[0]
      groundTruths[category] = cu.loadBoxIndexFile(dir + '/' + f)
  return groundTruths

def mapToGroundTruth(img, box, groundTruths):
  result = {}
  for cat in groundTruths.keys():
    try: annotations = groundTruths[cat][img]
    except: continue
    for gt in annotations:
      iou = det.IoU(box, gt)
      rel = findRelation(box, gt, iou)
      if rel != None:
        result[cat+'_'+rel] = iou
  #if len(result.keys()) > 1:
  #  print result
  return result

params = cu.loadParams('regionsFile groundTruthDir output')
regions,headers = loadRegionsFile(params['regionsFile'])
print 'Images:',len(regions)
groundTruths = loadAllGroundTruths(params['groundTruthDir'])
out = open(params['output'],'w')

catNames = groundTruths.keys()
relNames = ['_tight','_big','_inside']
catNames.sort()
categories = []
for r in relNames:
  for c in catNames:
    categories.append(c+r)
labels = dict([ (categories[i],i+1) for i in range(len(categories)) ])


dist = {}
counter = 0
for img in regions.keys():
  out.write('# ' + str(counter) + '\n')
  counter += 1
  for h in headers[img]:
    out.write(h[0] + '\n')
  samples = []
  for box in regions[img]:
    rels = mapToGroundTruth(img, box, groundTruths)
    for r in rels.keys():
      try: dist[r] += 1
      except: dist[r] = 1
      samples.append( [labels[r], '{:5.3f}'.format(rels[r])] + map(int,box) )
    if len(rels) == 0:
      samples.append( [0, 0.0,] + map(int,box) )
  out.write(str(len(samples)) + '\n')
  for s in samples:
    out.write(' '.join(map(str,s)) + '\n')
out.close()

keys = dist.keys()
keys.sort()
for k in keys:
  print k,dist[k]

