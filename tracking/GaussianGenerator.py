import sys
import TrajectorySimulator as trsim
import numpy as np
import pickle
import multiprocessing
import VisualAttention
import time
import VideoSequence
import os

SEQUENCE_LENGTH = 60
IMG_HEIGHT = 100
IMG_WIDTH = 100
TARGET_DIM = 4
MEMORY_SIZE = 4096 # Examples

# FUNCTION
# Make simulation of a single sequence given the simulator
def simulate(simulator, grayscale, computeFlow, start, end):
  data = []
  label = []
  simulator.start()
  for j, frame in enumerate(simulator):
    if grayscale:
      data += [np.asarray(frame.convert('L'))]
    else:
      data += [np.asarray(frame.convert('RGB'))]
    label += [simulator.getBox()]
  if computeFlow:
      flow = VisualAttention.computeFlowFromList(data[start:end])
  else:
      flow = None
  return data[start:end], label[start:end], flow

def wrapped_simulate(params):
    simulator, grayscale, computeFlow, start, end = params
    return simulate(simulator, grayscale, computeFlow, start, end)

## CLASS
## Simulator with Gaussian mixture models of movement
class GaussianGenerator(object):
    def __init__(self, imageDir, summaryPath, trajectoryModelSpec, cameraTrajectoryModelSpec, gmmPath, 
                 seqLength=60, imageSize=IMG_WIDTH, grayscale=True, parallel=True, numProcs=None, 
                 scenePathTemplate='images/train2014', objectPathTemplate='images/train2014', computeFlow=False):
        self.imageSize = imageSize
        self.seqLength = seqLength
        self.factory = None
        self.parallel = parallel
        if self.parallel:
            #Use maximum number of cores by default
            #self.numProcs =  multiprocessing.cpu_count() if numProcs is None or numProcs > multiprocessing.cpu_count() or numProcs < 1 else numProcs
            self.numProcs = numProcs
            self.results = None
            self.pool = None
        self.factory = trsim.SimulatorFactory(
            imageDir,
            trajectoryModelSpec,
            cameraTrajectoryModelSpec,
            summaryPath,
            gmmPath,
            scenePathTemplate=scenePathTemplate, objectPathTemplate=objectPathTemplate
            )
        self.grayscale = grayscale
        self.computeFlow = computeFlow
        # Setup replay memory
        channels = 3
        if self.grayscale: channels = 1
        if self.computeFlow: channels += 2
        self.memory = ReplayMemory(MEMORY_SIZE, seqLength, imageSize, channels)

#    def getSimulator(self):
#        simulator = self.factory.createInstance(camSize=(self.imageSize, self.imageSize))
#        return simulator
    def getSimulator(self):
       simulator = self.factory.createInstance(camSize=(self.imageSize, self.imageSize), maxSteps=self.seqLength)
       return simulator

    def initResults(self, batchSize):
        if self.grayscale:
            data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize), dtype=np.float32)
        else:
            #TODO: validate case of alpha channel: Not needed. When the image is read, it is forced to RGB
            data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize, 3), dtype=np.float32)
        if self.computeFlow:
            flow = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize, 2), dtype=np.float32)
        else:
            flow = None
        label = np.zeros((batchSize, self.seqLength, TARGET_DIM))
        return data, label, flow

    def getBatch(self, batchSize, start, end):
        if self.parallel:
            return self.getBatchInParallel(batchSize, start, end)
        else:
                data, label, flow = self.initResults(batchSize)
                for i in range(batchSize):
                    simulator = self.getSimulator()
                    simulator.start()
                    for j, frame in enumerate(simulator):
                        if self.grayscale:
                            data[i, j, :, :] = np.asarray(frame.convert('L'))
                        else:
                            data[i, j, :, :, :] = np.asarray(frame.convert('RGB'))
                        label[i, j] = simulator.getBox()
                data = data[:,start:end,...]
                label = label[:,start:end,...]
                if self.computeFlow:
                    flow = VisualAttention.computeFlowFromBatch(data)
                return data, label, flow

    def getBatchInParallelV0(self, batchSize, start, end):
        #TODO: avoid this condition by distributing and init end, but batchSize is needed and not available
        if self.results is None:
            #Distribute work for the first time
            self.results = self.distribute(batchSize, start, end)
        #Wait for results and collect them
        #import IPython
        #IPython.embed()
        auxResultsData = self.results.get(9999)
        data, label, flow = self.collect(batchSize, auxResultsData, start, end)
        self.memory.add(data, label, flow)
        #Distribute work
        self.results = self.distribute(batchSize, start, end)
        #Return previous results
        return data, label, flow

    def getBatchInParallel(self, batchSize, start, end):
        return self.getBatchFromMemory(batchSize, start, end)

    def getBatchFromMemory(self, batchSize, start, end):
        if not self.memory.dataAvailable(batchSize):
            while not self.memory.dataAvailable(batchSize):
                if self.results is None:
                    self.distributeAndCollect(batchSize, start, end)
                time.sleep(10)
        if self.memory.dataAvailable(batchSize) and self.results is None:
            self.distributeAndCollect(batchSize, start, end)
        return self.memory.getSample(batchSize)


    def distributeAndCollect(self, batchSize, start, end):
        self.initPool()
        self.results = "waiting"
        # Process simulations in parallel
        try:
            s = time.time()
            def callBack(result):
                data, label, flow = self.collect(batchSize, result, start, end)
                self.memory.add(data, label, flow)
                self.results = None
                print 'Elapsed:',(time.time()-s),'seg'
            results = self.pool.map_async(wrapped_simulate, [(self.getSimulator(), self.grayscale, self.computeFlow, start, end) for i in range(batchSize)], callback=callBack)
        except Exception as e:
            print 'Exception raised during map_async: {}'.format(e)
            self.pool.terminate()
            sys.exit()

    def initPool(self):
        # Lazy initialization
        if self.pool is None:
            self.pool = multiprocessing.Pool(self.numProcs)

    def distribute(self, batchSize, start, end):
        self.initPool()
        # Process simulations in parallel
        #try:
        if True:
            results = self.pool.map_async(wrapped_simulate, [(self.getSimulator(), self.grayscale, self.computeFlow, start, end) for i in range(batchSize)])
            return results
        #except Exception as e:
        else:
            print 'Exception raised during map_async: {}'.format(e)
            self.pool.terminate()
            sys.exit()

    def collect(self, batchSize, results, start, end):
        # Collect results and put them in the output
        index = 0
        data, label, flow = self.initResults(batchSize)
        data = data[:,start:end,...]
        label = label[:,start:end,...]
        if self.computeFlow: flow = flow[:,start:end,...]
        for frames, targets, of in results:
            if self.grayscale:
                data[index,:,:,:] = frames
            else:
                data[index,:,:,:,:] = frames
            if self.computeFlow:
                flow[index,:,:,:,:] = of
            label[index,:,:] = targets
            index += 1
            if index >= batchSize:
                break
        
        return data, label, flow

    def saveBatch(self, batchSize, start, end, outputDir, gtFilename='groundtruth.txt'):
        '''Generates a batch and saves it following VOT sequence structure'''
        data, label, flow = self.getBatch(batchSize, start, end)
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
        with open(os.path.join(outputDir, 'list.txt'), 'w') as listFile:
            for i in xrange(batchSize):
                videoSeq = VideoSequence.fromarray(data[i])
                videoSeq.addBoxes(label[i], outline='green')
                #TODO: improve sequence naming, hash instead of batch element
                videoSeq.exportToVideo(30, os.path.join(outputDir, str(i), 'video{}.mp4'.format(i)), keep=True)
                videoSeq.exportBoxes(os.path.join(outputDir, str(i), gtFilename), 'green')
                #TODO: check ending empty line
                listFile.write('{}\n'.format(i))

class ReplayMemory():

    def __init__(self, memorySize, frames, imageSize, channels):
        self.O = np.zeros( (memorySize, frames, imageSize, imageSize, channels), np.float32 )
        self.L = np.zeros( (memorySize, frames, 4), np.int )
        self.F = None
        self.pointer = 0
        self.usableRecords = 0

    def add(self, data, label, flow):
        batchSize = data.shape[0]
        if self.pointer < self.O.shape[0]-batchSize:
            self.pointer += batchSize
        else:
            self.pointer = 0

        self.O[self.pointer:self.pointer+batchSize,...] = data
        self.L[self.pointer:self.pointer+batchSize,...] = label

        if self.usableRecords < self.O.shape[0]:
            self.usableRecords += batchSize
        print "Added to memory:",self.pointer,self.usableRecords

    def getSample(self, batchSize):
        idx = np.arange(self.usableRecords)
        np.random.shuffle(idx)
        idx = idx[0:batchSize]
        O = self.O[idx,...]
        L = self.L[idx,...]
        F = self.F
        return O,L,F

    def dataAvailable(self, batchSize):
        if self.usableRecords >= 2*batchSize:
            return True
        else:
            return False

