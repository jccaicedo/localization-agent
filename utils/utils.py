import os,sys
import timeit
import numpy as np
import random
import pickle
import re
import MemoryUsage

floatType = np.float32
topHards = 5000
randomSeed = 0
tolerance = 1e-9
random.seed( randomSeed )
rnd = random

def loadParams(names):
  names = names.split()
  if len(sys.argv) < len(names)+1:
    print ' Parameters:',
    for p in names:
      print p,
    print ''
    sys.exit()
  else:
    params = {}
    for i in range(len(names)):
      params[names[i]] = sys.argv[i+1]
    return params

def tic():
  return timeit.default_timer()

def toc(msg, prev):
  curr = timeit.default_timer()
  print msg,"%.2f"%(curr-prev),'s'
  return tic()

def saveMatrix(matrix,outFile):
  outf = open(outFile,'w')
  np.savez_compressed(outf,matrix)
  outf.close()

def saveMatrixNoCompression(matrix,outFile):
  outf = open(outFile,'w')
  np.save(outf,matrix)
  outf.close()

def emptyMatrix(size):
  data = np.zeros(size)
  return data.astype(floatType)

def posOnes(size):
  return np.ones(size).astype(floatType)

def negOnes(size):
  return -1*posOnes(size)

def loadMatrix(filename):
  if os.path.isfile(filename):
    #print 'Single matrix file found'
    m = np.load( filename )
    op = m['arr_0']
    m.close()
    return op.astype(floatType)
  else:
    #print 'Multiple matrix files. Loading by parts' 
    i = 0
    next = filename.replace('.',str(i)+'.')
    #print next
    r = None
    while os.path.isfile(next):
      #print 'Part',i
      m = np.load(next)
      if r == None:
        r = m['arr_0']
      else:
        r = np.concatenate( (r,m['arr_0']) )
      i += 1
      m.close()
      next = filename.replace('.',str(i)+'.')
    #print 'Matrix',r.shape,'loaded'
    return r

def loadMatrixNoCompression(filename):
  if os.path.isfile(filename):
    m = np.load( filename )
    return m
  else:
    print 'No such file:',filename

def loadMatrixAndIndex(filename):
  m = loadMatrix(filename)
  idxFile = re.sub(r'\..+$',r'.idx',filename)
  l = [x.split() for x in open(idxFile)]
  return (m,l)

def saveModel(model,outFile):
  of = open(outFile,'w')
  of.write( pickle.dumps(model) )
  of.close()

def loadModel(modelFile):
  contents = open(modelFile).readlines()
  model = pickle.loads(''.join(contents))
  return model

def loadMatrixFromMultipleDirs(dirPrefix, filename, extension, index=False):
    suffix = '1'
    while dirPrefix.endswith('/'): dirPrefix = dirPrefix[:-1]
    fullPath = dirPrefix+suffix+'/'+filename+'.'+extension
    m = None
    while os.path.isfile(fullPath):
        q = loadMatrix(fullPath)
        suffix = str( int(suffix)+1 )
        fullPath = dirPrefix+suffix+'/'+filename+'.'+extension
        if m == None:
            m = q
        else:
            m = np.concatenate( (m,q) )
    return m

def loadBoxIndexFile(filename, idx=1):
  gt = [x.split() for x in open(filename)]
  images = {}
  for k in gt:
    try:
      images[k[0]] += [ map(float,k[idx:]) ]
    except:
      images[k[0]] = [ map(float,k[idx:]) ]
  return images

def mem(msg):
  print msg,'{:5.2f}'.format(MemoryUsage.memory()/(1024**3)),'GB'
