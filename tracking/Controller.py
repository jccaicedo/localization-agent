import argparse as AP
import time
import numpy as NP
import logging
import os
import sys

from RecurrentTracker import RecurrentTracker
from Validation import Validation
import Tester
import TheanoGruRnn
import GaussianGenerator
import VisualAttention

def clock(m, st): 
    print m,(time.time()-st)

class Controller(object):

    def train(self, tracker, epochs, batches, batchSize, generator, imgHeight, trackerModelPath, useReplayMem, generationBatchSize, seqLength):
        validation = Validation(8, batchSize, generator, imgHeight)
        for i in range(0, epochs):
            train_cost = 0
            et = time.time()
            for j in range(0, batches):

                # Obtain a batch of data to train on
                if not tracker.sampleFromMem():
                    st = time.time()
                    data, label = generator.getBatch(generationBatchSize)
                    # Keep max seqLength frames
                    if seqLength < data.shape[1]:
                        startFrame = NP.random.randint(0,data.shape[1]-seqLength)
                        data = data[:,startFrame:startFrame+seqLength,...]
                        label = label[:,startFrame:startFrame+seqLength,...]
                    # Mask the object that has to be tracked in the first frame
                    firstFrameMasks = VisualAttention.getSquaredMasks(data[:,0,...], label[:,0,:], 4, 0.1)
                    data[:,0,...] *= firstFrameMasks
                    # Replay memory management
                    storeInMem = (True and useReplayMem)  # When this flag is false, the memory is never used
                    # Normalize gray scale data
                    if generator.grayscale:
                        data = data[:, :, NP.newaxis, :, :]
                        data /= 255.0
                    # Center labels around zero, and scale them between [-1,1]
                    label = label / (imgHeight / 2.) - 1. 
                    clock('Simulations',st)
                else:
                    st = time.time()
                    data, label = tracker.getSample(generationBatchSize)
                    storeInMem = False
                    clock('No simulations',st)

                # Update parameters of the model
                st = time.time()                
                cost, bbox_seq = tracker.fit(data, label, storeInMem)
                clock('Training',st)
                
                print 'Cost', i, j, cost
                train_cost += cost
            bbox, iou = validation.validate(tracker)
            outputVideoDir = os.path.join(os.path.dirname(trackerModelPath), 'epoch'+str(i))
            if not os.path.exists(outputVideoDir):
                os.makedirs(outputVideoDir)
            logging.info('Saving validation videos for epoch %s in %s', i, outputVideoDir)
            #TODO: check postprocessing
            Tester.exportSequences(validation.valSet['data']*255.0, (validation.valSet['labels']+1)*validation.imgHeight/2., (bbox+1)*validation.imgHeight/2., grayscale, outputVideoDir)
            print 'Epoch average loss (train)', train_cost / (batches*batchSize)
            clock('Epoch time',et)
            TheanoGruRnn.saveModel(tracker.rnn, trackerModelPath)

class ControllerConfig(object):

    def __init__(self):
        self.parser = self.build_parser()
        self.args = self.parser.parse_args()

    def dump_args(self, configPath):
        with open(configPath, 'w') as configFile:
            for key, value in self.args.iteritems():
                configFile.write('{} {}\n'.format(key, value))

    def build_parser(self):
        parser = AP.ArgumentParser(description='Trains a RNN tracker', fromfile_prefix_chars='@')
        parser.add_argument('--imageDir', help='Root directory for images', type=str, default='/home/jccaicedo/data/coco')
        parser.add_argument('--summaryPath', help='Path of summary file', type=str, default='./cocoTrain2014Summary.pkl')
        parser.add_argument('--trajectoryModelPath', help='Trajectory model path', type=str, default='./gmmDenseAbsoluteNormalizedOOT.pkl')
        parser.add_argument('--epochs', help='Number of epochs with 32000 example sequences each', type=int, default=1)
        parser.add_argument('--generationBatchSize', help='Number of elements in one generation step', type=int, default=32)
        parser.add_argument('--batchSize', help='Number of elements in batch', type=int, default=32)
        parser.add_argument('--gpuBatchSize', help='Number of elements in GPU batch', type=int, default=4)
        parser.add_argument('--imgHeight', help='Image Height', type=int, default=224)
        parser.add_argument('--imgWidth', help='Image width', type=int, default=224)
        parser.add_argument('--gruStateDim', help='Dimension of GRU state', type=int, default=256)
        parser.add_argument('--seqLength', help='Length of sequences', type=int, default=60)
        parser.add_argument('--useReplayMem', help='Use replay memory to store simulated sequences', default=False, action='store_true')
        parser.add_argument('--convFilters', help='Number of filters to use in the convolutional layer', type=int, default=32)
        #TODO: Check default values or make required
        parser.add_argument('--trackerModelPath', help='Name of model file', type=str, default='model.pkl')
        parser.add_argument('--caffeRoot', help='Root of Caffe dir', type=str, default='/home/jccaicedo/caffe/')
        parser.add_argument('--cnnModelPath', help='Name of model file', type=str, default='/home/jccaicedo/data/simulations/cnns/googlenet/bvlc_googlenet.caffemodel')
        parser.add_argument('--deployPath', help='Path to Protobuf deploy file for the network', type=str, default='/home/jccaicedo/data/simulations/cnns/googlenet/deploy.prototxt')
        parser.add_argument('--zeroTailFc', help='', type=bool, default=False)
        parser.add_argument('--meanImage', help='Path to mean image for ImageNet dataset relative to Caffe', default='python/caffe/imagenet/ilsvrc_2012_mean.npy')
        parser.add_argument('--layerKey', help='Key string of layer name to use as features', type=str, default='inception_5b/output')
        parser.add_argument('--learningRate', help='SGD learning rate', type=float, default=0.0005)
        parser.add_argument('--useCUDNN', help='Use CUDA CONV or THEANO', type=bool, default=False)
        parser.add_argument('--modelArch', help='Network architecture', type=str, default='base', choices=['base', 'caffe', 'lasagne', 'twoConvLayers','threeConvLayers'])
        parser.add_argument('--sequential', help='Make sequential simulations', default=False, action='store_true')
        parser.add_argument('--numProcs', help='Number of processes for parallel simulations', type=int, default=None)
        #TODO: Evaluate specifying the level instead if more than debug is needed   
        parser.add_argument('--debug', help='Enable debug logging', default=False, action='store_true')
        parser.add_argument('--norm', help='Norm type for cost', default=TheanoGruRnn.l2.func_name, choices=[TheanoGruRnn.smooth_l1.func_name, TheanoGruRnn.l2.func_name])
        parser.add_argument('--useAttention', help='Enable attention', type=str, default='no', choices=['no', 'gaussian', 'square'])
        
        return parser

if __name__ == '__main__':
    
    # Configuration
    
    config = ControllerConfig()
    globals().update(vars(config.args))
    
    logging.BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(funcName)s:%(lineno)d:%(message)s'
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    #TODO: make arguments not redundant
    if modelArch == 'caffe':
        from CaffeCnn import CaffeCnn
        #Make batch size divisible by gpuBatchSize to enable reshaping
        batchSize = int(batchSize/gpuBatchSize)*gpuBatchSize
        logging.debug('Batch size: %s GPU batch size: %s', batchSize, gpuBatchSize)
        #Make generation batch size divisible by batchSize
        generationBatchSize = int(generationBatchSize/batchSize)*batchSize
        logging.debug('Generation batch size: %s GPU batch size: %s', generationBatchSize, gpuBatchSize)
        cnn = CaffeCnn(imgHeight, imgWidth, deployPath, cnnModelPath, caffeRoot, seqLength, meanImage, layerKey, gpuBatchSize)
        gruInputDim = reduce(lambda a,b: a*b, cnn.outputShape()[-3:])
        grayscale = False
    elif modelArch == 'lasagne':
        cnn = gruInputDim = None
        grayscale = False
    elif modelArch == 'base':
        cnn = gruInputDim = None
        imgHeight = imgWidth = 100
        grayscale = True
    elif modelArch == 'twoConvLayers':
        cnn = gruInputDim = None
        imgHeight = imgWidth = 128
        grayscale = False
    elif modelArch == 'threeConvLayers':
        cnn = gruInputDim = None
        imgHeight = imgWidth = 192
        grayscale = False

    #Avoid maximum recursion limit exception when pickling by increasing limit from ~1000 by default
    sys.setrecursionlimit(10000)
    
    try:
        rnn = TheanoGruRnn.loadModel(trackerModelPath)
    except Exception as e:
        #TODO: silent for trax
        print 'Exception loading model from {}: {}'.format(trackerModelPath, e)
        print 'Creating new model'
        rnn = TheanoGruRnn.TheanoGruRnn(gruInputDim, gruStateDim, GaussianGenerator.TARGET_DIM, batchSize, seqLength, zeroTailFc, learningRate, useCUDNN, imgHeight, modelArch, getattr(TheanoGruRnn, norm), useAttention, modelPath=cnnModelPath, layerKey=layerKey, convFilters=convFilters)
    
    
    tracker = RecurrentTracker(cnn, rnn)
    
    generator = GaussianGenerator.GaussianGenerator(imageDir, summaryPath, trajectoryModelPath, seqLength=60, imageSize=imgHeight, grayscale=grayscale, parallel=not sequential, numProcs=numProcs)
    
    controller = Controller()
    M = 9600 # Constant number of example sequences per epoch
    batches = M/batchSize
    try:
        controller.train(tracker, epochs, batches, batchSize, generator, imgHeight, trackerModelPath, useReplayMem, generationBatchSize, seqLength)
    #TODO: evaluate if it is wise to save on any exception
    except KeyboardInterrupt:
        TheanoGruRnn.saveModel(tracker.rnn, trackerModelPath)
