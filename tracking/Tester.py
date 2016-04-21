import numpy as NP
import VideoSequence as SQ
import matplotlib.pyplot as PLT
import os
import logging

from PIL import Image

"""
Scale the bounding boxes from a frame dimensions to another

@type bboxes: numpy.array(frames, coordinates)
@param bboxes: The array with the bounding boxes, one per row
@type fromSize: [height, width]
@param fromSize: The dimensions of the source frames
@type toSize: [height, width]
@param toSize: The dimensions of the target frames

@rtype: numpy.array(frames, coordinates)
@return: The array with the bounding boxes, one per row
"""
def scaleBboxes(bboxes, fromSize, toSize):
    scaledFromSize = 1.0 / NP.array(fromSize)
    coords = bboxes.shape[1]
    scaledFromSize = NP.multiply(scaledFromSize, NP.ones((2, coords / 2))).flatten()
    scaledBboxes = NP.multiply(bboxes, scaledFromSize)
    toScaledSize = NP.multiply(toSize, NP.ones((2, coords / 2))).flatten()
    return NP.multiply(scaledBboxes, toScaledSize)

def getIntOverUnion(bboxTruth, bboxPred):
    #TODO: what if boxes are not referenced/ordered according to distance from 0,0 in screen?
    left = NP.max([bboxPred[..., 0], bboxTruth[..., 0]], axis=0)
    top = NP.max([bboxPred[..., 1], bboxTruth[..., 1]], axis=0)
    right = NP.min([bboxPred[..., 2], bboxTruth[..., 2]], axis=0)
    bottom = NP.min([bboxPred[..., 3], bboxTruth[..., 3]], axis=0)
    intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
    label_area = NP.abs(bboxTruth[..., 2] - bboxTruth[..., 0]) * NP.abs(bboxTruth[..., 3] - bboxTruth[..., 1])
    predict_area = NP.abs(bboxPred[..., 2] - bboxPred[..., 0]) * NP.abs(bboxPred[..., 3] - bboxPred[..., 1])
    union = label_area + predict_area - intersect
    iou = intersect / union
    
    return iou

def padBatch(array, batchSize):
    size = array.shape[0]
    padNum = batchSize - size % batchSize
    paddedArray = NP.pad(array, [(0,padNum)]+[(0,0)]*len(array.shape[1:]), 'constant')
    return paddedArray
    
def getFrames(frames, isGrayScale):
    fs, _, _ = frames.shape
    
    for i in range(fs):
        image = Image.fromarray(frames[i, :, :])
            
        if(isGrayScale):
            image = image.convert("RGB")
        
        yield image
        
def exportSequences(frames, gtBoxes, predBoxes, isGrayScale, outputVideoDir):
    seqs, fs, _, _, _ = frames.shape
    fps = 30
    
    for i in range(seqs):
        seqFs = getFrames(frames[i, :, 0, :, :], isGrayScale)
        sq = SQ.VideoSequence(seqFs)
        sq.addBoxes(gtBoxes[i, :, :], "green")
        sq.addBoxes(predBoxes[i, :, :], "red")
        output = os.path.join(outputVideoDir, 'sequence' + str(i) + ".mp4")
        sq.exportToVideo(fps, output)

class Tester(object):
    
    tracker = None
    
    def __init__(self, tracker):
        self.tracker = tracker
        
        
    def test(self, data, label, flow, batchSize, imageHeight, withVideoGen, seqLength, targetDim, outputVideoDir):
        size = data.shape[0]
        iters = size / batchSize + (size % batchSize > 0)
        bboxSeqTest = NP.empty((0, seqLength, targetDim))
        
        data = padBatch(data, batchSize)
        label = padBatch(label, batchSize)
        
        for i in range(1, iters + 1):
            start = batchSize * (i-1)
            end = batchSize * i
            dataTest = data[start:end, ...]
            labelTest = label[start:end, ...]
            if flow is None:
                flowTest = None
            else:
                flowTest = flow[start:end, ...]
            pred = self.tracker.forward(dataTest, labelTest, flowTest)
            bboxSeqTest = NP.append(bboxSeqTest, pred, axis=0)
        
        data = data[0:size, ...]
        bboxSeqTest = bboxSeqTest[0:size, ...]
        label = label[0:size, ...]
        
        if(withVideoGen):
            #TODO: use postprocessed data for data
            exportSequences(data, label , bboxSeqTest, outputVideoDir)
        
        iou = getIntOverUnion(label, bboxSeqTest)

        return iou, bboxSeqTest
    
            
    def plotOverlapMeasures(self, iouMeasures, title, xLabel, yLabel, outputPath):
        for idx, iou in enumerate(iouMeasures):
            fig = PLT.figure(figsize=(20, 15))
            PLT.plot(iou, label="Sequence #" + str(idx))
            PLT.title(title)
            PLT.xlabel(xLabel)
            PLT.ylabel(yLabel)
            PLT.legend(loc='upper left', bbox_to_anchor=(0.8, 0.8))
            figPath = os.path.join(outputPath, )
            PLT.savefig(outputPath + "Sequence_" + str(idx) + "_IOU.png")
            PLT.close(fig)
            PLT.clf()
            
            
    def plotGeneralMeasures(self, iouMeasures, title, outputPath):
        measures = {}
        measures["mean"] = NP.mean(iouMeasures, axis=0)
        measures["max"] = NP.max(iouMeasures, axis=0)
        measures["min"] = NP.min(iouMeasures, axis=0)
        measures["median"] = NP.median(iouMeasures, axis=0)
        measures["std"] = NP.std(iouMeasures, axis=0)
        
        for name, measure in measures.iteritems():
            fig = PLT.figure(figsize=(20, 15))
            PLT.plot(measure, label=name)
            PLT.legend(loc='upper left', bbox_to_anchor=(0.8, 0.8))
            PLT.title(title)
            PLT.xlabel("Frame")
            PLT.ylabel(name)
            figPath = os.path.join(outputPath, name + ".png")
            PLT.savefig(figPath)
            PLT.close(fig)
            PLT.clf()
        measures = {}
        
        measures["all"] =  iou
        measures["mean"] = NP.mean(iou, axis=0)
        measures["max"] = NP.max(iou, axis=0)
        measures["min"] = NP.min(iou, axis=0)
        measures["median"] = NP.median(iou, axis=0)
        measures["std"] = NP.std(iou, axis=0)
        
        return measures
            
    
    def getTestData(self, generator, batchSize, imageHeight):
        dataTest, labelTest = generator.getBatch(batchSize)
        
        if generator.grayscale:
            dataTest = dataTest[:, :, NP.newaxis, :, :]
        
        dataTest /= 255.0
        labelTest = labelTest / (imageHeight / 2.) - 1.
        
        return dataTest, labelTest
    
    def loadImageSequence(self, sequenceDir, newShape=None, extension='.jpg', relativeGtPath='groundtruth.txt'):
        framePaths = sorted([framePath for framePath in os.listdir(sequenceDir) if framePath.endswith(extension)])
        #Get the original image size
        originalSize = Image.open(os.path.join(sequenceDir,framePaths[0])).size 
        frames = (Image.open(os.path.join(sequenceDir, framePath)) for framePath in framePaths)
        data = NP.array([NP.asarray(frame if newShape is None else frame.resize(newShape)) for frame in frames])
        with open(os.path.join(sequenceDir, relativeGtPath), 'r') as gtFile:
            gtLines = gtFile.readlines()
        #TODO: better polygon handling
        posCorrection = NP.array(newShape if newShape is not None else originalSize, dtype=NP.float)/NP.array(originalSize)
        gtPolygons = NP.array([map(float, line.strip().split(',')) for line in gtLines])
        labels = gtPolygons[:, [0,1,4,5]]*[posCorrection[0], posCorrection[1], posCorrection[0], posCorrection[1]]
        return data, labels

import socket
import pickle
import VideoSequenceData as vs
import argparse as ap

#Toy server that returns the initial box
def modelServer(modelPath):
    logging.BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(funcName)s:%(lineno)d:%(message)s'
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    fileHandler = logging.FileHandler(os.path.join(os.path.dirname(modelPath), 'traxServer.log'))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.DEBUG)
    with open(modelPath, 'r') as modelFile:
        model = pickle.load(modelFile)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 2501))
    s.listen(5)
    while True:
        conn, addr = s.accept()
        logging.debug('Connected by %s', addr)
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    logging.debug('No more data')
                    break
                try:
                    box = eval(data)
                    conn.send(str(box))
                except SyntaxError as e:
                    if os.path.exists(data):
                        logging.debug('Path to frame exists %s', data)
                        conn.send(str(box))
                    else:
                        raise e                    
        finally:
            conn.close()
        
def modelClient(serverPort, libvotPath):
    logging.BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(process)d:%(funcName)s:%(lineno)d:%(message)s'
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    fileHandler = logging.FileHandler(os.path.join('/home/fmpaezri/repos/localization-agent', 'traxClient.log'))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.DEBUG)
    tcw = vs.TraxClientWrapper(libvotPath)
    initBox = tcw.getBox()
    hasNext = True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', serverPort))
    s.sendall(str(initBox))
    repliedBox = s.recv(4096)
    logging.info('Received initial box %s', repliedBox)
    logging.info('Box as list %s', eval(repliedBox))
    while hasNext:
        frame = tcw.path
        s.sendall(frame)
        data = s.recv(4096)
        logging.info('Received new box: %s', data)
        tcw.reportBox(eval(data))
        hasNext = tcw.nextStep()
    s.close()
    
if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Script for evaluation')
    parser.add_argument('testType', help='Type of test to perform', default='trax', choices=['trax', 'custom'])
    parser.add_argument('serverPort', help='Port of listening server', type=int, default=2501)
    parser.add_argument('--libvotPath', help='Path to libvot library for TRAX integration', default='/home/fmpaezri/repos/vot-toolkit/tracker/examples/native/libvot.so')
    args = parser.parse_args()

    if args.testType == 'trax':
        modelClient(args.serverPort, args.libvotPath)
    else:
        raise Exception('Not implemented yet')

