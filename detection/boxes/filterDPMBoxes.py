import os,sys
import libDetection as det
import masks as mk

if len(sys.argv) < 3:
  print 'Use: filterDPMBoxes.py input output'
  sys.exit()


boxes = mk.loadBoxIndexFile(sys.argv[1])
totalBoxes = 0
keptBoxes = 0
out = open(sys.argv[2],'w')
for img in boxes.keys():
  unique = set([':'.join(map(str,map(int,x))) for x in boxes[img]])
  for b in boxes[img]:
    boxHash = ':'.join(map(str,map(int,b)))
    if boxHash in unique:
      unique.remove(boxHash)
    else:
      continue
    totalBoxes += 1
    a = det.area(b)
    w,h = b[2]-b[0],b[3]-b[1]
    if a > 400 and w > 0 and h > 0:
      out.write(img + ' ' + ' '.join(map(str,map(int,b))) + '\n' )
      keptBoxes += 1

print 'Total boxes:',totalBoxes,'Filtered:',totalBoxes-keptBoxes,'Final Boxes:',keptBoxes
out.close()
