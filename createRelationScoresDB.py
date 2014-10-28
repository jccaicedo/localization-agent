import os,sys
import utils as cu

params = cu.loadParams('scoresDirectory outputFile')

## READ DIRECTORY CONTENTS
categories = []
records = []
for f in os.listdir(params['scoresDirectory']):
  names = f.split('_')[0:2]
  cat = '_'.join(names)
  categories.append( cat )
  data = [x.split() for x in open(params['scoresDirectory']+'/'+f)]
  for d in data:
    records.append( [d[0]] + d[2:6] + [cat, d[1]] )
categories.sort()
categories = dict([ (categories[i],i) for i in range(len(categories)) ])
print 'Categories:',len(categories)
print 'Records:',len(records)

## ORGANIZE RECORDS BY BOX
boxes = {}
for r in records:
  key = ' '.join(r[0:5])
  try: boxes[key][ categories[r[5]] ] = r[6]
  except: boxes[key] = {categories[r[5]]:r[6]}
print 'Unique boxes:',len(boxes)

## WRITE BOXES WITH SPARSE DETECTIONS
out = open(params['outputFile'],'w')
for b in boxes.keys():
  out.write(b)
  cat = boxes[b].keys()
  cat.sort()
  for c in cat:
    out.write(' ' + str(c) + ':' + boxes[b][c])
  out.write('\n')
out.close()

## WRITE INDEX OF CATEGORIES
out = open(params['outputFile']+'.categories','w')
cat = categories.keys()
cat.sort()
for c in cat:
  out.write(str(categories[c]) + ' ' + c + '\n')
out.close()
