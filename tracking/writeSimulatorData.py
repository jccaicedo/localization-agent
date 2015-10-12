import h5py
import numpy as np
import time
import VideoSequenceData as vsd
import os,sys
from multiprocessing import Process, JoinableQueue, Queue

filePath = vsd.dataDir + 'simulation.hdf5' # Output filename
GEN = 10 # Number of generator objects
SIM = 10 # Simulations per generator

# FUNCTION
# Distributed work in multiple cores
def worker(inQueue, outQueue, simulations, output):
  for data in iter(inQueue.get,'stop'):
    index,sequenceGenerator = data
    results = []
    for i in range(simulations):
      r = simulate(sequenceGenerator)
      results.append(r)
    outQueue.put(results)
    inQueue.task_done()
  inQueue.task_done()
  return True

# FUNCTION
# Coordination of multiple workers and their results
def processData(sequenceGenerators, simulations, output):
  numProcs = min(len(sequenceGenerators),10) # max number of cores: 10
  taskQueue = JoinableQueue()
  resultQueue = Queue()
  processes = []
  # Start workers
  for i in range(numProcs):
    t = Process(target=worker, args=(taskQueue, resultQueue, simulations, output))
    t.daemon = True
    t.start()
    processes.append(t)

  # Assign tasks to workers
  i = 0
  for gen in sequenceGenerators:
    taskQueue.put( (i,gen) )
    i += simulations
  for i in range(len(processes)):
    taskQueue.put('stop')

  # Collect results and send them to an HDF5 file
  index = 0
  for k in range(numProcs):
    data = resultQueue.get()
    for frames,targets in data:
      output.create_dataset("frames"+str(index),data=frames)
      output.create_dataset("targets"+str(index),data=targets)
      index += 1

# FUNCTION
# Simulation of one sequence
def simulate(seq):
  seq.prepareSequence()
  # Store in a numpy array
  simFrames = np.zeros((vsd.totalFrames,vsd.channels,vsd.imgSize,vsd.imgSize))
  simTargets = np.zeros((vsd.totalFrames,4))
  step = 0
  while seq.nextStep():
    simFrames[step,:,:,:] = seq.getFrame()
    simTargets[step,:] = seq.getMove()
    step += 1
  return (simFrames,simTargets)

# USE
# Main Procedure
if __name__ == '__main__':

  generators = [vsd.VideoSequenceData() for i in range(GEN)]
  while True:
    outFile = h5py.File(filePath,'w')
    processData(generators, SIM, outFile)
    outFile.close()

    while os.path.exists(filePath):
      time.sleep(0.5)

