import os,sys
import libDetection as det

if len(sys.argv) < 5:
  print 'Use: selectTopDetections.py scoresFile overlapFile maxDetections output'
  sys.exit()

scores = [x.split() for x in open(sys.argv[1])]
overlaps = [x.split() for x in open(sys.argv[2])]
maxDet = int(sys.argv[3])
outFile = sys.argv[4]

print 'Total records => Scores:',len(scores),'Overlaps:',len(overlaps)

scoresImages = {}
for s in scores:
  try: scoresImages[s[0]].append( s[1:] )
  except: scoresImages[s[0]] = [ s[1:] ]
for img in scoresImages.keys():
  scoresImages[img].sort(key=lambda x: float(x[0]), reverse=True)
  limit = min(len(scoresImages[img]), maxDet)
  scoresImages[img] = scoresImages[img][0:limit]

overlapsImages = {}
for o in overlaps:
  try: overlapsImages[o[0]].append( o[1:] )
  except: overlapsImages[o[0]] = [ o[1:] ]

print 'Total images => Scores:',len(scoresImages),'Overlaps:',len(overlapsImages)

perfectMatches = 0
selectedRecords = {}
for img in scoresImages.keys():
  selectedRecords[img] = []
  for i in range(len(scoresImages[img])):
    b1 = map(float, scoresImages[img][i][1:5])
    maxIoU = 0.0
    bestMatch = -1
    for j in range(len(overlapsImages[img])):
      b2 = map(float, overlapsImages[img][j][0:4])
      iou = det.IoU(b1,b2)
      if iou > maxIoU:
        bestMatch = j
        maxIoU = iou
    if maxIoU == 1.0: perfectMatches += 1
    selectedRecords[img].append( scoresImages[img][i][1:5] +  
        [scoresImages[img][i][0]] + overlapsImages[img][bestMatch][-2:] )
    del overlapsImages[img][bestMatch]

print 'Perfect matches:',perfectMatches

out = open(outFile, 'w')
for img in selectedRecords.keys():
  for r in selectedRecords[img]:
    out.write(img + ' ' + ' '.join(r) + '\n' )
out.close()
