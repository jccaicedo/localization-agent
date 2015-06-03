import logging
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import utils.libDetection as libDet
import learn.rl.RLConfig as config
import utils.utils as cu
import sequence

def parse_gt(gtPath, style='corners'):
    '''
    Parses a line oriented bounding box description (whitespace or comma separated)
    '''
    #some file are comma-separated instead of tab-separated
    gtFile = open(gtPath)
    gt = numpy.array([map(float, line.strip().replace(',', '\t').split()) for line in gtFile])
    #benchmark uses xo,yo,w,h instead of 'corners' x0,y0,x1,y1
    if style == 'corners':
        #add x0,y0 to w,h
        gt[:,2:] += gt[:,:2]
    gtFile.close()
    logging.debug('Found {} lines in ground truth file {}'.format(len(gt), gtPath))
    return gt

def plot_gt(gt, formatString='', style='corners'):
    '''
    Plots the different columns of a parsed bounding box sequence
    '''
    #TODO: review x,y axes convention and change labels
    if style == 'corners':
        expectedLabels = ['X lower left coord.', 'Y lower left coord.', 'X upper right coord.', 'Y upper right coord.']
    elif style == 'dims':
        expectedLabels = ['X lower left coord.', 'Y lower left coord.', 'Width', 'Height']
    if not len(expectedLabels) == gt.shape[1]:
        raise Exception('Label and column mismatch ({} vs. {})'.format(len(expectedLabels), gt.shape[1]))
    for column in xrange(len(expectedLabels)):
        plt.subplot(2,2,column+1)
        plt.plot(numpy.linspace(0, 1, num=gt.shape[0]), gt[:,column], formatString, label=expectedLabels[column])
        plt.xlabel('Frame index')
        plt.ylabel('Pixel')
        plt.title(expectedLabels[column])
    plt.show()

def dataset_gt(gtsDir, pattern='groundtruth_rect.txt'):
    '''
    Parses the bounding boxes of the dataset, where each folder is a sequence and groundtruth file has a name pattern
    '''
    sequenceNames = os.listdir(gtsDir)
    gtsDict = {}
    for sequenceName in sequenceNames:
        #TODO: make adaptable with a reg-exp as pattern
        gtPath = os.path.join(gtsDir, sequenceName, pattern)
        if not os.path.exists(gtPath):
            logging.debug('Ommiting sequence {} as ground truth file {} wasnt found'.format(sequenceName, pattern))
            continue
        sequenceGt = parse_gt(gtPath)
        gtsDict[sequenceName] = sequenceGt
        logging.debug('Parsed gt for sequence {}'.format(sequenceName))
    return gtsDict

def measure_inertia(bbSequence, measure):
    '''Applies an inertia mesure over consecutive pairs of gt using "corners" style'''
    inertia = numpy.zeros((bbSequence.shape[0]-1,1))
    bbDiag = bbSequence.copy()
    for index in xrange(len(inertia)):
        inertia[index] = measure(bbDiag[index], bbDiag[index+1])
    return inertia

def plot_inertias(gtsDict, measure, subplots):
    index = 1
    rows, columns = subplots
    for key in gtsDict:
        if index < rows*columns:
            plt.subplot(rows, columns, index)
            inertia = measure_inertia(gtsDict[key], measure=measure)
            plt.plot(numpy.linspace(0,1, inertia.shape[0]), inertia)
            plt.xticks([0,1])
            plt.yticks([0,1])
            plt.title(key)
            index += 1

#taken from TrackerTask
def center(box):
  return [ (box[2] + box[0])/2.0 , (box[3] + box[1])/2.0 ]

def euclideanDist(c1, c2):
  return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2

PLACE_LANDMARK = 8

def landmark_tracking(configPath):
    '''Generates a dictionary mapping sequence names to list of tracked boxes'''
    config.readConfiguration(configPath)
    imageSuffix = config.get('frameSuffix')
    sequenceDir = config.get('sequenceDir')
    testMemoryDir = config.get('testMemory')
    seqDatabasePath = config.get('testDatabase')
    seqDatabase = [x.strip() for x in open(seqDatabasePath, 'r')]

    trackingResults = {}

    for sequenceName in seqDatabase:
        seqName, seqSpan, seqStart, seqEnd = cu.parseSequenceSpec(sequenceName)
        imageDir = os.path.join(sequenceDir, seqName, config.get('imageDir'))
        gtPath = os.path.join(sequenceDir, seqName, config.get('gtFile'))
        aSequence = sequence.fromdir(imageDir, gtPath, suffix=imageSuffix)

        #no seqSpan means full sequence
        #frames start at 2 in list, but include 0 in gt
        if seqSpan is None:
            start = 1
            end = len(aSequence.frames)+1
        else:
            start = int(seqStart)
            end = int(seqEnd)
            if start < 1 or end >= len(aSequence.frames) or start > end:
                raise ValueError('Start {} or end {} outside of bounds {}-{}'.format(start, end, 1, len(aSequence.frames)))

        if sequenceName not in trackingResults:
            trackingResults[sequenceName] = []
        
        for frameIndex in range(start, end):
            testMemoryPath = os.path.join(testMemoryDir, seqName, config.get('imageDir'), '{:04d}{}'.format(frameIndex, '.txt'))
            if os.path.exists(testMemoryPath):
                testMemory = cu.load_memory(testMemoryPath) 
                if PLACE_LANDMARK in testMemory['actions']:
                    landmarkIndex = testMemory['actions'].index(PLACE_LANDMARK)
                    trackingResults[sequenceName].append(testMemory['boxes'][landmarkIndex])
                else:
                    trackingResults[sequenceName].append(trackingResults[sequenceName][-1])
            else:
                trackingResults[sequenceName].append(aSequence.boxes[start-1])
    
    return trackingResults

def performance_metrics(configPath):
    trackingResults = landmark_tracking(configPath)
    config.readConfiguration(configPath)
    imageSuffix = config.get('frameSuffix')
    sequenceDir = config.get('sequenceDir')
    seqDatabasePath = config.get('testDatabase')
    seqDatabase = [x.strip() for x in open(seqDatabasePath, 'r')]

    dists = {}
    ious = {}

    for sequenceName in seqDatabase:
        seqName, seqSpan, seqStart, seqEnd = cu.parseSequenceSpec(sequenceName)
        imageDir = os.path.join(sequenceDir, seqName, config.get('imageDir'))
        gtPath = os.path.join(sequenceDir, seqName, config.get('gtFile'))
        aSequence = sequence.fromdir(imageDir, gtPath, suffix=imageSuffix)

        #no seqSpan means full sequence
        #frames start at 2 in list, but include 0 in gt
        if seqSpan is None:
            start = 1
            end = len(aSequence.frames)+1
        else:
            start = int(seqStart)
            end = int(seqEnd)
            if start < 1 or end >= len(aSequence.frames) or start > end:
                raise ValueError('Start {} or end {} outside of bounds {}-{}'.format(start, end, 1, len(aSequence.frames)))

        if sequenceName not in trackingResults:
            ious[sequenceName] = []
            dists[sequenceName] = []
        
        ious[sequenceName] = map(libDet.IoU, aSequence.boxes[start-1:end-1], trackingResults[sequenceName])
        gtCenters = map(center, aSequence.boxes[start-1:end-1])
        trackingCenters = map(center, trackingResults[sequenceName])
        dists[sequenceName] = numpy.sqrt(map(euclideanDist, gtCenters, trackingCenters))

    return dists, ious

def performance_plots(configPath, plot=False):
    dists, ious = performance_metrics(configPath)
    
    assert dists.keys() == ious.keys(), 'Different sequences in performance dictionaries'

    #TODO; make parameterizable
    steps = 10
    iouThresholds = numpy.linspace(0,1,steps,endpoint=True)
    distThresholds = numpy.linspace(1,50,steps,endpoint=True)

    distsPercents = numpy.zeros((len(dists.keys()), steps))
    iousPercents = numpy.zeros((len(ious.keys()), steps))

    for seqIndex, seqName in enumerate(dists.keys()):
        for thresholdIndex in xrange(steps):
            distsPercents[seqIndex, thresholdIndex] = numpy.less_equal(dists[seqName], distThresholds[thresholdIndex]).mean()
            iousPercents[seqIndex, thresholdIndex] = numpy.greater_equal(ious[seqName], iouThresholds[thresholdIndex]).mean()
    
    if plot:
        plt.subplot(1,2,1)
        plt.plot(distThresholds, distsPercents.mean(axis=0))
        plt.title('Precision')
        plt.subplot(1,2,2)
        plt.plot(iouThresholds, iousPercents.mean(axis=0))
        plt.title('Success')
        plt.show()

    return distsPercents, iousPercents
