import numpy as NP
import VideoSequence as SQ

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


class Tester(object):
    
    tracker = None
    
    def __init__(self, tracker):
        self.tracker = tracker
        
        
    def test(self, data, label, batchSize, imageHeight, grayscale, withVideoGen):
        size = data.shape[0]
        iters = size / batchSize + (size % batchSize > 0)
        bboxSeqTest = NP.empty((0, 60, 4))
        
        data, label = self.preprocessData(data, label, imageHeight, grayscale, batchSize)
        
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
            self.exportSequences(data * 255.0, (label + 1) * imageHeight / 2., (bboxSeqTest + 1) * imageHeight / 2., grayscale)
        
        iou = getIntOverUnion(label, bboxSeqTest)
        
        measures = {}
        
        measures["all"] =  iou
        measures["mean"] = NP.mean(iou, axis=0)
        measures["max"] = NP.max(iou, axis=0)
        measures["min"] = NP.min(iou, axis=0)
        measures["median"] = NP.median(iou, axis=0)
        measures["std"] = NP.std(iou, axis=0)
        
        return measures
        
    def exportSequences(self, frames, gtBoxes, predBoxes, isGrayScale):
        seqs, fs, _, _, _ = frames.shape
        fps = 30
        
        for i in range(seqs):
            seqFs = self.getFrames(frames[i, :, 0, :, :], isGrayScale)
            sq = SQ.VideoSequence(seqFs)
            sq.addBoxes(gtBoxes[i, :, :], "red")
            sq.addBoxes(predBoxes[i, :, :], "blue")
            output = "/home/fhdiaze/Data/video" + str(i) + ".mp4"
            sq.exportToVideo(fps, output)
            
    
    def getTestData(self, generator, batchSize, imageHeight):
        dataTest, labelTest = generator.getBatch(batchSize)
        
        if generator.grayscale:
            dataTest = dataTest[:, :, NP.newaxis, :, :]
        
        dataTest /= 255.0
        labelTest = labelTest / (imageHeight / 2.) - 1.
        
        return dataTest, labelTest

    def preprocessData(self, data, label, imageHeight, grayscale, batchSize):
        size = data.shape[0]
        padNum = batchSize - size % batchSize
        
        if grayscale:
            data = data[:, :, NP.newaxis, :, :]
        
        data = NP.pad(data, ((0,padNum), (0,0), (0,0), (0,0), (0,0)), 'constant')
        label = NP.pad(label, ((0,padNum), (0,0), (0,0)), 'constant')
        
        data /= 255.0
        label = label / (imageHeight / 2.) - 1.
        
        return data, label
    
    def getFrames(self, frames, isGrayScale):
        fs, _, _ = frames.shape
        
        for i in range(fs):
            image = Image.fromarray(frames[i, :, :])
            
            if(isGrayScale):
                image = image.convert("RGB")
            
            yield image