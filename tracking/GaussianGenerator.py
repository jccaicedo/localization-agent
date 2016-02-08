import sys
import TrajectorySimulator as trsim
import numpy as np
import pickle

COCO_DIR = '/data1/mscoco/'

import multiprocessing

SEQUENCE_LENGTH = 60
IMG_HEIGHT = 100
IMG_WIDTH = 100

# FUNCTION
# Make simulation of a single sequence given the simulator
def simulate(simulator, grayscale):
  data = []
  label = []
  simulator.start()
  for j, frame in enumerate(simulator):
    if grayscale:
      data += [np.asarray(frame.convert('L'))]
    else:
      data += [np.asarray(frame.convert('RGB'))]
    label += [simulator.getBox()]
  return data, label

def wrapped_simulate(params):
    simulator, grayscale = params
    return simulate(simulator, grayscale)

## CLASS
## Simulator with Gaussian mixture models of movement
class GaussianGenerator(object):
    def __init__(self, seqLength=60, imageSize=IMG_WIDTH, dataDir='.', grayscale=True, single=True, parallel=True, numProcs=None, summaryName='/cocoTrain2014Summary.pkl'):
        self.imageSize = imageSize
        self.seqLength = seqLength
        trajectoryModelPath = dataDir + '/gmmDenseAbsoluteNormalizedOOT.pkl'
        self.factory = None
        self.parallel = parallel
        if self.parallel:
            #Use maximum number of cores by default
            self.numProcs =  multiprocessing.cpu_count() if numProcs is None or numProcs > multiprocessing.cpu_count() or numProcs < 1 else numProcs
            self.results = None
            self.pool = None
        if not single:
            # Generates a factory to create random simulator instances
            self.factory = trsim.SimulatorFactory(
                COCO_DIR,
                trajectoryModelPath=trajectoryModelPath,
                summaryPath = dataDir + summaryName,
                scenePathTemplate='images/train2014', objectPathTemplate='images/train2014'
                )
        modelFile = open(trajectoryModelPath, 'r')
        self.trajectoryModel = pickle.load(modelFile)
        modelFile.close()
        self.grayscale = grayscale

    def getSimulator(self):
        if self.factory is None:
            simulator = self.getSingleSimulator()
        else:
            simulator = self.factory.createInstance(camSize=(self.imageSize, self.imageSize))
        return simulator

    def getSingleSimulator(self):
        scenePath = COCO_DIR + "/images/train2014/COCO_train2014_000000011826.jpg"
        objectPath = COCO_DIR + "/images/train2014/COCO_train2014_000000250067.jpg"
        polygon = [618.23, 490.13, 615.76, 488.48, 612.89, 488.48, 610.42, 491.36, 609.19, 494.65, 607.54, 498.35, 607.13, 503.29, 606.72, 510.28, 610.42, 512.33, 612.89, 513.15, 616.18, 513.98, 619.88, 513.57, 621.93, 510.28, 623.58, 506.58, 623.58, 503.7, 623.16, 500.0, 621.93, 496.71, 619.46, 493.42]
        simulator = trsim.TrajectorySimulator(scenePath, objectPath, polygon=polygon, trajectoryModel=self.trajectoryModel, camSize=(self.imageSize, self.imageSize))
        
        return simulator

    def initResults(self, batchSize):
        if self.grayscale:
            data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize), dtype=np.float32)
        else:
            #TODO: validate case of alpha channel
            data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize, 3), dtype=np.float32)
        label = np.zeros((batchSize, self.seqLength, 4))
        return data, label

    def getBatch(self, batchSize):
        if self.parallel:
            return self.getBatchInParallel(batchSize)
        else:
                data, label = self.initResults(batchSize)
                for i in range(batchSize):
                    simulator = self.getSimulator()
                    simulator.start()
                    for j, frame in enumerate(simulator):
                        if self.grayscale:
                            data[i, j, :, :] = np.asarray(frame.convert('L'))
                        else:
                            data[i, j, :, :, :] = np.asarray(frame.convert('RGB'))
                        label[i, j] = simulator.getBox()
                        
                return data, label

    def getBatchInParallel(self, batchSize):
        #TODO: avoid this condition by distributing and init end, but batchSize is needed and not available
        if self.results is None:
            #Distribute work for the first time
            self.results = self.distribute(batchSize)
        #Wait for results and collect them
        data, label = self.collect(batchSize, self.results.get(9999))
        #Distribute work
        self.results = self.distribute(batchSize)
        #Return previous results
        return data, label

    def initPool(self):
        # Lazy initialization
        if self.pool is None:
            self.pool = multiprocessing.Pool(self.numProcs)

    def distribute(self, batchSize):
        self.initPool()

        # Process simulations in parallel
        try:
            results = self.pool.map_async(wrapped_simulate, [(self.getSimulator(), self.grayscale) for i in range(batchSize)])
            return results
        except Exception as e:
            print 'Exception raised during map_async: {}'.format(e)
            self.pool.terminate()
            sys.exit()

    def collect(self, batchSize, results):
        # Collect results and put them in the output
        index = 0
        data, label = self.initResults(batchSize)
        for frames, targets in results:
            if self.grayscale:
                data[index,:,:,:] = frames
            else:
                data[index,:,:,:,:] = frames
            label[index,:,:] = targets
            index += 1
        
        return data, label
