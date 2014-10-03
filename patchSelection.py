import os,sys
import libDetection as det
import utils as cu
import Image

def shiftBox(box, limitX, limitY):
  if box[0] < 0:
    box[2] -= box[0]
    box[0] = 0
  if box[1] < 0:
    box[3] -= box[1]
    box[1] = 0
  if box[2] >= limitX:
    box[2] = limitX - 1
  if box[3] >= limitY:
    box[3] = limitY - 1
  w,h = box[2]-box[0],box[3]-box[1]
  if w < h:
    box[3] -= h - w
  elif h < w:
    box[2] -= w - h
  return box

def getBestFitBoxes(gt, w, h, cropSize):
  scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
  patchSizes = [cropSize/s for s in scales]

  bw,bh = gt[2]-gt[0],gt[3]-gt[1]
  largerAxis = max(bw,bh) 
  targetSize = min(w,h)
  scaleBoxes = []
  for i in range(len(patchSizes)-1):
    if patchSizes[i] >= largerAxis and largerAxis > patchSizes[i+1]:
      targetSize = patchSizes[i]
  if largerAxis < patchSizes[-1]:
    targetSize = patchSizes[-1]
  if targetSize < min(w,h):
    xMargin = (targetSize - bw)/2
    yMargin = (targetSize - bh)/2
    nb = [gt[0] - xMargin, gt[1] - yMargin, gt[2] + xMargin, gt[3] + yMargin]
    scaleBoxes.append( shiftBox(nb, w, h) )
  else:
    adjustedSize = min(w,h) - 1
    if bw > bh:
      y1,y2 = 0,adjustedSize
      parts = int( round( bw/adjustedSize + 0.3) )
      step = (bw - adjustedSize)/max(parts-1,1)
      for i in range(parts):
        nb = [gt[0] + i*step, y1, gt[0] + i*step + adjustedSize, y2]
        scaleBoxes.append( shiftBox(nb, w, h)  )
    else:
      x1,x2 = 0,adjustedSize
      parts = int( round( bh/adjustedSize + 0.3) )
      step = (bh - adjustedSize)/max(parts-1,1)
      for i in range(parts):
        nb = [x1, gt[1] + i*step, x2, gt[1] + i*step + adjustedSize]
        scaleBoxes.append( shiftBox(nb, w, h)  )
  return scaleBoxes

if __name__ == "__main__":
  params = cu.loadParams("groundTruthBoxes imageDir outputDir cropSize")
  groundTruthBoxes = cu.loadBoxIndexFile(params['groundTruthBoxes'])
  cropSize = int(params['cropSize'])
  projections = {}
  overlaps = []
  ious = []

  for img in groundTruthBoxes.keys():
    print img
    name = img.split('/')[1]
    if not os.path.isfile(params['imageDir'] + '/' + name + '.JPEG'): continue
    im = Image.open(params['imageDir'] + '/' + name + '.JPEG')
    w,h = im.size
    try: p = projections[img]
    except: projections[img] = {}
    for gt in groundTruthBoxes[img]:
      scaleBoxes = getBestFitBoxes(gt, w, h, cropSize)
      for b in scaleBoxes:
        b = map(int,b)
        projections[img][ ' '.join(map(str,b[0:4])) ] = 1 # add the best box to the index
        #im.crop(b[0:4]).save(params['outputDir']+img+'_'+'_'.join(map(str,b[0:4])) + '_pr.jpg')
        #im.crop(map(int,gt)).save(params['outputDir']+img+'_GT_'+'_'.join(map(str,map(int,gt))) + '.jpg')
      
  out = open(params['outputDir'] + '/boxes_file.txt','w')
  for img in projections.keys():
    for box in projections[img]:
      out.write(img + ' ' + box + '\n')
      if projections[img][box] != 1:
        print 'Box',box,'in image',img,'covers',projections[img][box],'objects'
      
  out.close()
