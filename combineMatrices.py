import os,sys
import utils as cu
import numpy as np

params = cu.loadParams('matrix1 matrix2 output')
Ma,Ia = cu.loadMatrixAndIndex(params['matrix1'])
Mb,Ib = cu.loadMatrixAndIndex(params['matrix2'])

extension = params['matrix1'].split('.')[-1]
cu.saveMatrix( np.concatenate( (Ma,Mb) ) , params['output']+'.'+extension)
out = open(params['output']+'.idx','w')
for r in Ia:
  out.write(' '.join(r)+'\n')
for r in Ib:
  out.write(' '.join(r)+'\n')
out.close()

