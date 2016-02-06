import sys
#sys.path.insert(0, r'/home/fhdiaze/Code/localization-agent/tracking/')
import TrajectorySimulator as trsim
import numpy as np
import pickle

COCO_DIR = '/home/jccaicedo/data/coco'

import time
import os
from multiprocessing import Process, JoinableQueue, Queue
import multiprocessing

SEQUENCE_LENGTH = 60
IMG_HEIGHT = 100
IMG_WIDTH = 100

# FUNCTION
# Distribute work in multiple cores
def worker(inQueue, outQueue, simulations):
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
# Make simulation of a single sequence given the simulator
def simulate(simulator):
  data = np.zeros((SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH))
  label = np.zeros((SEQUENCE_LENGTH, 4))
  simulator.start()
  for j, frame in enumerate(simulator):
    data[j, :, :] = np.asarray(frame.convert('L'))
    label[j, :] = simulator.getBox()              
  return data, label


## CLASS
## Simulator with Gaussian mixture models of movement
class GaussianGenerator(object):
    def __init__(self, seqLength=60, imageSize=IMG_WIDTH, dataDir='.'):
        self.imageSize = imageSize
        self.seqLength = seqLength
        trajectoryModelPath = dataDir + '/gmmDenseAbsoluteNormalizedOOT.pkl'
        # Generates a factory to create random simulator instances
        self.factory = trsim.SimulatorFactory(
            COCO_DIR, 
            trajectoryModelPath=trajectoryModelPath, 
            summaryPath = dataDir + '/cocoTrain2014Summary.pkl', 
            scenePathTemplate='images/train2014', objectPathTemplate='images/train2014'
            )
        modelFile = open(trajectoryModelPath, 'r')
        self.trajectoryModel = pickle.load(modelFile)
        modelFile.close()

    def getSimulator(self):
        emptyPolygon = True
        
        while emptyPolygon:
            simulator = self.factory.createInstance(drawBox=False, camera=True, drawCam=False, cameraContentTransforms=None, camSize=(self.imageSize, self.imageSize))
            emptyPolygon = len(simulator.polygon) == 0
            
        return simulator
    
    def getSingleSimulator(self):
        scenePath = COCO_DIR + "/images/train2014/COCO_train2014_000000011826.jpg"
        objectPath = COCO_DIR + "/images/train2014/COCO_train2014_000000250067.jpg"
        polygon = [618.23, 490.13, 615.76, 488.48, 612.89, 488.48, 610.42, 491.36, 609.19, 494.65, 607.54, 498.35, 607.13, 503.29, 606.72, 510.28, 610.42, 512.33, 612.89, 513.15, 616.18, 513.98, 619.88, 513.57, 621.93, 510.28, 623.58, 506.58, 623.58, 503.7, 623.16, 500.0, 621.93, 496.71, 619.46, 493.42]
        simulator = trsim.TrajectorySimulator(scenePath, objectPath, polygon=polygon, trajectoryModel=self.trajectoryModel, camSize=(self.imageSize, self.imageSize))
        
        return simulator

    def getBatch(self, batchSize):
        data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize), dtype=np.float32)
        label = np.zeros((batchSize, self.seqLength, 4))
        for i in range(batchSize):
            simulator = self.getSingleSimulator() 
            simulator.start()
            for j, frame in enumerate(simulator):
                data[i, j, :, :] = np.asarray(frame.convert('L'))
                label[i, j] = simulator.getBox()
                
        return data, label

    def getBatchInParallel(self, batchSize):
        numProcs =  multiprocessing.cpu_count() # max number of cores
        if batchSize % numProcs != 0:
            print "Please use a multiple of",numProcs,"for the batch size"
            sys.exit()
        simulations = batchSize/numProcs
        taskQueue = JoinableQueue()
        resultQueue = Queue()
        processes = []
        # Start workers
        for i in range(numProcs):
            t = Process(target=worker, args=(taskQueue, resultQueue, simulations))
            t.daemon = True
            t.start()
            processes.append(t)

        # Assign tasks to workers
        i = 0
        for gen in [self.getSingleSimulator() for i in range(numProcs)]:
            taskQueue.put( (i,gen) )
            i += simulations
        for i in range(len(processes)):
            taskQueue.put('stop')

        # Collect results and put them in the output
        index = 0
        data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize), dtype=np.float32)
        label = np.zeros((batchSize, self.seqLength, 4))
        for k in range(numProcs):
            result = resultQueue.get()
            for frames,targets in result:
                data[index,:,:,:] = frames
                label[index,:,:] = targets
                index += 1
        
        return data, label


