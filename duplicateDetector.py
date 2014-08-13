import os,sys
from PIL import Image
import utils as cu
import numpy as np

def findNearestNeighbor(H,i):
  J = np.tile(H[i,:],(H.shape[0],1))
  R = np.sum( np.abs(J-H), axis=1 )
  R[i] = np.inf
  return np.argmin(R),np.min(R)

params = cu.loadParams('imageDir')
dir = params['imageDir']
allImages = os.listdir(dir)
H = np.zeros( (len(allImages),768) )
print 'Scanning',len(allImages),'images'
imgs = 0
for f in allImages:
  try:
    im = Image.open(dir+'/'+f)
    h = np.asarray(im.histogram())
    if len(h) == 256:
      H[imgs,:] = np.tile(np.asarray(h),(1,3))
    else:
      H[imgs,:] = np.asarray(h)
    imgs += 1
  except:
    print 'Problems with',f
  
H = H[0:imgs,:]
for i in range(imgs):
  j,similarity = findNearestNeighbor(H,i)
  print allImages[i],allImages[j],similarity
