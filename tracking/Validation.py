import numpy as NP
import VisualAttention

class Validation(object):

    def __init__(self, valBatches, batchSize, generator, imgHeight):
        self.valBatches = valBatches
        self.batchSize = batchSize
        # Create a fixed validation set for this training session
        d, l = generator.getBatch(batchSize)
        dataShape = [valBatches*d.shape[0]] + list(d.shape[1:])
        labelShape = [valBatches*l.shape[0]] + list(l.shape[1:])
        self.valSet = {'data':NP.zeros(dataShape),'labels':NP.zeros(labelShape)}
        self.valSet['data'][0:batchSize,...] = d
        self.valSet['labels'][0:batchSize,...] = l
        for i in range(1,valBatches):
            d, l = generator.getBatch(batchSize)
            ffm = VisualAttention.getSquaredMasks(d[:,0,...], l[:,0,:], 4, 0.1)
            d[:,0,...] *= ffm
            start = i*batchSize
            end = (i+1)*batchSize
            self.valSet['data'][start:end,...] = d
            self.valSet['labels'][start:end,...] = l
        if generator.grayscale:
            self.valSet['data'] = self.valSet['data'][:, :, NP.newaxis, :, :]
            self.valSet['data'] /= 255.0
        self.valSet['labels'] = self.valSet['labels'] / (imgHeight / 2) - 1
        print 'Validation set ready with examples:',  self.valSet['data'].shape

    
    def validate(self, tracker):
        # Get predictions
        L = self.valSet['labels']
        D = self.valSet['data']
        bbox = NP.zeros( (self.valBatches*self.batchSize, L.shape[1], L.shape[2]) )
        for i in range(self.valBatches):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            bbox[start:end,...] = tracker.forward(D[start:end,...], L[start:end,...])
        # Compute IoU (TODO: Move this code to a reusable function in another place)
        left = NP.max([bbox[:, :, 0], L[:, :, 0]], axis=0)
        top = NP.max([bbox[:, :, 1], L[:, :, 1]], axis=0)
        right = NP.min([bbox[:, :, 2], L[:, :, 2]], axis=0)
        bottom = NP.min([bbox[:, :, 3], L[:, :, 3]], axis=0)
        intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
        label_area = (L[:, :, 2] - L[:, :, 0]) * (L[:, :, 2] - L[:, :, 0] > 0) * (L[:, :, 3] - L[:, :, 1]) * (L[:, :, 3] - L[:, :, 1] > 0)
        predict_area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 2] - bbox[:, :, 0] > 0) * (bbox[:, :, 3] - bbox[:, :, 1]) * (bbox[:, :, 3] - bbox[:, :, 1] > 0)
        union = label_area + predict_area - intersect
        iou = intersect / union
        # Report to the log
        print 'IoU in validation set:',NP.average(iou)
         
