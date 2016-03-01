import sys
import TrajectorySimulator as trsim
import numpy as np
import pickle
import multiprocessing

SEQUENCE_LENGTH = 60
IMG_HEIGHT = 100
IMG_WIDTH = 100
TARGET_DIM = 4

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
    def __init__(self, imageDir, summaryPath, trajectoryModelPath, seqLength=60, imageSize=IMG_WIDTH, grayscale=True, parallel=True, numProcs=None, scenePathTemplate='images/train2014', objectPathTemplate='images/train2014'):
        self.imageSize = imageSize
        self.seqLength = seqLength
        self.factory = None
        self.parallel = parallel
        if self.parallel:
            #Use maximum number of cores by default
            self.numProcs =  multiprocessing.cpu_count() if numProcs is None or numProcs > multiprocessing.cpu_count() or numProcs < 1 else numProcs
            self.results = None
            self.pool = None
        self.factory = trsim.SimulatorFactory(
            imageDir,
            trajectoryModelPath=trajectoryModelPath,
            summaryPath = summaryPath,
            scenePathTemplate=scenePathTemplate, objectPathTemplate=objectPathTemplate
            )
        self.grayscale = grayscale

    def getSimulator(self):
        simulator = self.factory.createInstance(camSize=(self.imageSize, self.imageSize))
        return simulator

    def initResults(self, batchSize):
        if self.grayscale:
            data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize), dtype=np.float32)
        else:
            #TODO: validate case of alpha channel
            data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize, 3), dtype=np.float32)
        label = np.zeros((batchSize, self.seqLength, TARGET_DIM))
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
