import os,sys
import utils as cu
import scipy.io

if __name__ == "__main__":
  params = cu.loadParams('matFilesDir outFile')
  out = open(params['outFile'],'w')
  counter = 0
  for f in os.listdir(params['matFilesDir']):
    if not f.endswith('.mat') or f == 'gt_pos_layer_5_cache.mat': continue
    img = f.replace('.mat','')
    counter += 1
    print counter,img
    mat = scipy.io.loadmat(params['matFilesDir'] + '/' + f)
    idx = mat['gt'] == 0
    mat['boxes'] = mat['boxes'][idx[:,0],:]
    for i in range(mat['boxes'].shape[0]):
      box = mat['boxes'][i,:].tolist()
      out.write(img + ' ' + ' '.join(map(str, map(int,box))) + '\n' )
  out.close()
