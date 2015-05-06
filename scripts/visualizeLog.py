import os,sys
import Image, ImageDraw
import random

if len(sys.argv) < 2:
  print 'Use: visualizeLog.py logFile'
  sys.exit()

imgDir = '/home/caicedo/data/allimgs/'

log = [x.replace(',','').replace('[','').replace(']','').split() for x in open(sys.argv[1])]
log = [x for x in log if len(x) > 10]
random.shuffle(log)

counter = 0
for l in log:
  if float(l[11]) < 0: continue
  box1 = map(int, map(float, l[1:5]))
  box2 = map(int, map(float, l[6:10]))
  im = Image.open(imgDir + l[0].replace("'",'') + '.jpg')
  draw = ImageDraw.Draw(im)
  draw.rectangle(box1, outline='#0F0')
  draw.rectangle(box2, outline=128)
  draw.text(box1[0:2], l[11])
  draw.text(box1[2:4], l[10])
  im.show()
  counter += 1
  if counter > 10:
    break
