import os,sys
import utils as cu
import libDetection as det
import numpy as np

if __name__ == "__main__":
  params = cu.loadParams('boxesFile minSize outputFile')
  boxes = [x.split() for x in open(params['boxesFile'])]
  minA = float(params['minSize'])**2
  images = {}
  found = set()
  for box in boxes:
    a = det.area( map(int,box[1:]) )
    if a >= minA:
      try:
        images[ box[0] ].append(box[1:])
      except:
        images[ box[0] ] = [box[1:]]
    found.add(box[0])
  # Add records for images that do not have enough area to comply with the filter      
  missing = found.symmetric_difference( images.keys() )
  missing = [b for b in boxes if b[0] in missing]
  allAreas = {}
  for m in missing:
      img = m[0]
      try:
          allAreas[img]['box'].append( m[1:] )
          allAreas[img]['area'].append( det.area( map(int,m[1:]) ) )
      except:
          allAreas[img] = { 'box':[m[1:]], 'area':[det.area( map(int,m[1:]) )] }
  for img in allAreas.keys():
      idx = np.argsort(allAreas[img]['area'])
      images[img] = [allAreas[img]['box'][i] for i in idx[-5:]]
  # Write output    
  out = open(params['outputFile'],'w')
  for k in images.keys():
    for b in images[k]:
      out.write(k+' '+' '.join(b)+'\n')
  out.close()
