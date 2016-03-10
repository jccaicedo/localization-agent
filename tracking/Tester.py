import numpy as NP
import VideoSequence as SQ
import matplotlib.pyplot as PLT
import os

from PIL import Image

def getIntOverUnion(bboxTruth, bboxPred):
    left = NP.max([bboxPred[:, :, 0], bboxTruth[:, :, 0]], axis=0)
    top = NP.max([bboxPred[:, :, 1], bboxTruth[:, :, 1]], axis=0)
    right = NP.min([bboxPred[:, :, 2], bboxTruth[:, :, 2]], axis=0)
    bottom = NP.min([bboxPred[:, :, 3], bboxTruth[:, :, 3]], axis=0)
    intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
    label_area = (bboxTruth[:, :, 2] - bboxTruth[:, :, 0]) * (bboxTruth[:, :, 2] - bboxTruth[:, :, 0] > 0) * (bboxTruth[:, :, 3] - bboxTruth[:, :, 1]) * (bboxTruth[:, :, 3] - bboxTruth[:, :, 1] > 0)
    predict_area = (bboxPred[:, :, 2] - bboxPred[:, :, 0]) * (bboxPred[:, :, 2] - bboxPred[:, :, 0] > 0) * (bboxPred[:, :, 3] - bboxPred[:, :, 1]) * (bboxPred[:, :, 3] - bboxPred[:, :, 1] > 0)
    union = label_area + predict_area - intersect
    iou = intersect / union
    
    return iou

def preprocessData(data, label, imageHeight, grayscale, batchSize):
    size = data.shape[0]
    padNum = batchSize - size % batchSize
        
    if grayscale:
        data = data[:, :, NP.newaxis, :, :]
        
    data = NP.pad(data, ((0,padNum), (0,0), (0,0), (0,0), (0,0)), 'constant')
    label = NP.pad(label, ((0,padNum), (0,0), (0,0)), 'constant')
        
    data /= 255.0
    label = label / (imageHeight / 2.) - 1.
        
    return data, label
    
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
        
        
    def test(self, data, label, batchSize, imageHeight, grayscale, withVideoGen, seqLength, targetDim, outputVideoDir):
        size = data.shape[0]
        iters = size / batchSize + (size % batchSize > 0)
        bboxSeqTest = NP.empty((0, seqLength, targetDim))
        
        data, label = preprocessData(data, label, imageHeight, grayscale, batchSize)
        
        for i in range(1, iters + 1):
            start = batchSize * (i-1)
            end = batchSize * i
            dataTest = data[start:end, ...]
            labelTest = label[start:end, ...]
            pred = self.tracker.forward(dataTest, labelTest)
            bboxSeqTest = NP.append(bboxSeqTest, pred, axis=0)
        
        data = data[0:size, ...]
        bboxSeqTest = bboxSeqTest[0:size, ...]
        label = label[0:size, ...]
        
        if(withVideoGen):
            exportSequences(data * 255.0, (label + 1) * imageHeight / 2., (bboxSeqTest + 1) * imageHeight / 2., grayscale, outputVideoDir)
        
        iou = getIntOverUnion(label, bboxSeqTest)

        return iou
    
            
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
    
    def loadImageSequence(self, sequenceDir, extension='.jpg', relativeGtPath='groundtruth.txt'):
        framePaths = sorted([framePath for framePath in os.listdir(sequenceDir) if framePath.endswith(extension)])
        data = (numpy.asarray(Image.open(os.path.join(sequenceDir, framePath))) for framePath in framePaths)
        with open(os.path.join(sequenceDir, relativeGtPath), 'r') as gtFile:
            gtLines = gtFile.readlines()
        #TODO: better polygon handling
        gtPolygons = numpy.array([map(float, line.strip().split(',')) for line in gtLines])
        labels = gtPolygons[:, [0,1,4,5]]
        return data, labels