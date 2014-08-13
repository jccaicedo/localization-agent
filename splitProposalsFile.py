import os,sys
import math, re

if len(sys.argv) < 4:
  print 'Use: splitProposalsFile.py bboxesFile parts output'
  sys.exit()

bboxes = [x for x in open(sys.argv[1])]
parts  = int(sys.argv[2])
outFile = sys.argv[3] + '/part.'

##################################
# Organize boxes by source image
##################################
images = {}
r = re.compile(r"^([^\s]+)")
for box in bboxes:
  name = r.match(box).group()
  try:
    images[ name ].append(box)
  except:
    images[ name ] = [box]
print 'File parsed'

##################################
# Write chuncks of files
##################################
imagesPerPart = math.ceil( float(len(images.keys())) / float(parts) )
print 'Preparing parts with',imagesPerPart,'images'
counter = 0
part = 0
currentFile = open( outFile+str(part), 'w' )
for i in images.keys():
  for box in images[i]:
    currentFile.write(box)
  counter += 1
  if counter == imagesPerPart:
    currentFile.close()
    part += 1
    counter = 0
    currentFile = open( outFile+str(part), 'w' )
print 'Total parts:',part
currentFile.close()
