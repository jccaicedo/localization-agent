import os,sys
import utils as cu
import scipy.io
import numpy as np
import libDetection as det

def loadBoxIndexFile(filename):
  gt = [x.split() for x in open(filename)]
  images = {}
  for k in gt:
    try:
      images[k[0]] += [ [k[1]] + map(float,k[2:]) ]
    except:
      images[k[0]] = [ [k[1]] + map(float,k[2:]) ]
  return images

def getCategories():
  cat = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
  categories = {}
  id = 0
  for c in cat:
    categories[c + '_big'] = id
    id += 1
    categories[c + '_inside'] = id
    id += 1
  return categories

if __name__ == "__main__":
  params = cu.loadParams('relationsAnnotations matFilesDir outDir')
  relations = loadBoxIndexFile(params['relationsAnnotations'])
  print 'Relations loaded'
  categories = getCategories()
  counter = 0
  for f in os.listdir(params['matFilesDir']):
    if not f.endswith('.mat') or f == 'gt_pos_layer_5_cache.mat': continue
    counter += 1
    if os.path.isfile(params['outDir'] + '/' + f): continue
    img = f.replace('.mat','')
    print counter,img
    mat = scipy.io.loadmat(params['matFilesDir'] + '/' + f)
    idx = mat['gt'] == 0
    mat['feat'] = mat['feat'][idx[:,0],:]
    mat['gt'] = mat['gt'][idx[:,0],:]
    mat['boxes'] = mat['boxes'][idx[:,0],:]
    mat['overlap'] = np.zeros( (mat['feat'].shape[0], len(categories)) )
    mat['class'] = np.zeros( (mat['feat'].shape[0], 1) )
    duplicate = []
    try:
      groundTruths = relations[img]
    except:
      groundTruths = []
    for gt in groundTruths:
      for i in range(mat['boxes'].shape[0]):
        box = mat['boxes'][i,:].tolist()
        iou = det.IoU(box, gt[1:])
        if iou > mat['overlap'][i, categories[gt[0]]]:
          mat['overlap'][i, categories[gt[0]]] = iou
        if iou == 1:
          mat['gt'][i] = 1.0
          if mat['class'][i] == 0 or mat['class'][i] == categories[gt[0]]+1:
            mat['class'][i] = categories[gt[0]]+1
          else:
            duplicate.append( {'row':i, 'class':categories[gt[0]]+1} )
    shift = 0
    for d in duplicate:
      for key in ['feat','gt','boxes','overlap','class']:
        mat[key] = np.vstack( (mat[key][d['row'] + shift,:], mat[key]) )
      shift += 1
      mat['class'][0] = d['class']
    scipy.io.savemat(params['outDir'] + '/' + f, mat, do_compression=True)
