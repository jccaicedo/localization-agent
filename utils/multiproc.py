import os,sys
from multiprocessing import Process, JoinableQueue, Queue

def worker(inQueue, outQueue, task):
  for data in iter(inQueue.get,'stop'):
    result = task(data)
    if result != None:
      outQueue.put( result )
    else:
      outQueue.put('Ignore')
    inQueue.task_done()
  inQueue.task_done()
  return True

def processData(dataList, task, numProcs):
  taskQueue = JoinableQueue()
  resultQueue = Queue()
  processes = []
  for i in range(numProcs):
    t = Process(target=worker, args=(taskQueue, resultQueue, task))
    t.daemon = True
    t.start()
    processes.append(t)

  for d in dataList:
    taskQueue.put( d )
  for i in range(len(processes)):
    taskQueue.put('stop')

  results = []
  retrieved = 0
  while retrieved < len(dataList):
    data = resultQueue.get()
    retrieved += 1
    if data != 'Ignore':
      results += data
  return results


