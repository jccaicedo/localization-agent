import os,sys
import numpy as np
from multiprocessing import Process, JoinableQueue, Queue
from StringIO import StringIO
import re
from utils import floatType,tic,toc

def loadFeaturesAndIndex(img,content,index):
  try:
    data = np.load(StringIO(content))
    m = data['arr_0']
    l = [x.split() for x in index.split('\n') if x != '']
    data.close()
  except:
    print 'Error with',img
  return (m.astype(floatType),l)

def loadFilteredFeaturesAndIndex(img,content,index):
  import libDetection as det
  import numpy as np
  MAX_BOXES = 1000
  m,l = loadFeaturesAndIndex(img,content,index)
  if len(l) > MAX_BOXES:
    limit = max(len(l)-MAX_BOXES,0)
    areas = [ det.area(map(float,b[1:])) for b in l]
    areasIdx = np.argsort(areas)
    candidates = np.asarray([i >= limit for i in areasIdx])
    boxes = [l[i] for i in areasIdx if i >= limit]
    return (m[candidates].astype(floatType),boxes)
  else:
    return m,l

def worker(inQueue, outQueue, task):
  for data in iter(inQueue.get,'stop'):
    #s = tic()
    img,content,index = data
    #img,con,ind = data
    #conFile = open(con)
    #idxFile = open(ind)
    #content = conFile.read()
    #index = idxFile.read()
    features,bboxes = loadFeaturesAndIndex(img,content,index)
    result = task.run(img,features,bboxes)
    if result != None:
      outQueue.put(result)
    else: 
      outQueue.put('Ignore')
      #print img,'did not produce any result.'
    inQueue.task_done()
    #idxFile.close()
    #conFile.close()
    #toc(img,s)
  inQueue.task_done()
  return True

def processData(imageList,featuresDir,featuresExt,task):
  numProcs = 8
  taskQueue = JoinableQueue()
  resultQueue = Queue()
  processes = []
  for i in range(numProcs):
    t = Process(target=worker, args=(taskQueue, resultQueue, task))
    t.daemon = True
    t.start()
    processes.append(t)

  for img in imageList:
    filename = featuresDir+'/'+img+'.'+featuresExt
    idxFile = re.sub(r'\..+$',r'.idx',filename)
    content = open(filename)
    index = open(idxFile)
    taskQueue.put( (img,content.read(),index.read()) )
    #taskQueue.put( (img,filename,idxFile) )
    index.close()
    content.close()
  for i in range(len(processes)):
    taskQueue.put('stop')

  results = []
  retrieved = 0
  while retrieved < len(imageList):
    data = resultQueue.get()
    retrieved += 1
    if data != 'Ignore':
      results.append(data)
  return results

