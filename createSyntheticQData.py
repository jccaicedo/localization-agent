import os,sys
import utils as cu
import Image

params = cu.loadParams('positiveBoxes negativeBoxes imgDir output')

pos = [x.split() for x in open(params['positiveBoxes'])]
neg = [x.split() for x in open(params['negativeBoxes'])]

boxes = {}
for p in pos:
  try: boxes[p[0]].append(p[1:] + [0])
  except: boxes[p[0]] = [p[1:] + [0]]

for n in neg:
  try: boxes[n[0]].append(n[1:] + [1])
  except: boxes[n[0]] = [p[1:] + [1]]

flipLabel = lambda x: 0 if x == 1 else 1
counter = 0
out = open(params['output'],'w')
for img in boxes.keys():
  out.write('# ' + str(counter) + '\n')
  imPath = params['imgDir'] + '/' + img + '.jpg'
  im = Image.open(imPath)
  w,h = im.size
  out.write(imPath + '\n3\n' + str(w) + '\n' + str(h) + '\n' + str(2*len(boxes[img])) + '\n')
  for b in boxes[img]:
    out.write(str(b[-1]) + ' 1.0 0.0 ' + ' '.join(b[0:4]) + '\n' )
    out.write(str(flipLabel(b[-1])) + ' -1.0 0.0 ' + ' '.join(b[0:4]) + '\n' )
  counter += 1
out.close()
