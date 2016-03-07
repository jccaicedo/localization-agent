import numpy as NP
import VisualAttention
import Tester
import h5py

class Validation(object):

    def __init__(self, valBatches, batchSize, generator, imgHeight, computeFlow, seqLen, saveData=False):
        self.valBatches = valBatches
        self.batchSize = batchSize
        self.imgHeight = imgHeight
        self.computeFlow = computeFlow
        self.seqLength = seqLen
        filepath = '_valset_' + str(imgHeight) + '.h5'
        try:
            # Try loading a preexisting validation set
            h5f = h5py.File(filepath, 'r')
            self.valSet = {'data':h5f['data'][:], 'labels':h5f['labels'][:]}
            print 'Loaded validation set with examples:',  self.valSet['data'].shape
        except:
            # Create a fixed validation set for this training session
            d, l = generator.getBatch(batchSize)
            dataShape = [valBatches*d.shape[0]] + list(d.shape[1:])
            labelShape = [valBatches*l.shape[0]] + list(l.shape[1:])
            self.valSet = {'data':NP.zeros(dataShape),'labels':NP.zeros(labelShape)}
            self.valSet['data'][0:batchSize,...] = d
            self.valSet['labels'][0:batchSize,...] = l
            for i in range(1,valBatches):
                d, l = generator.getBatch(batchSize)
                start = i*batchSize
                end = (i+1)*batchSize
                self.valSet['data'][start:end,...] = d
                self.valSet['labels'][start:end,...] = l
            if generator.grayscale:
                self.valSet['data'] = self.valSet['data'][:, :, NP.newaxis, :, :]
                self.valSet['data'] /= 255.0
            print 'Generated validation set with examples:',  self.valSet['data'].shape
            if saveData:
                h5f = h5py.File(filepath, "w")
                dset = h5f.create_dataset("data", data=self.valSet['data'])
                lset = h5f.create_dataset("labels", data=self.valSet['labels'])
                h5f.close()
                print 'Validation file saved'
        self.extractFlow()
    
    def validate(self, tracker):
        # Get predictions
        L = self.valSet['labels'][:,0:self.seqLength,...]
        D = self.valSet['data'][:,0:self.seqLength,...]
        if self.computeFlow:
            F = self.valSet['flow']
        bbox = NP.zeros( (self.valBatches*self.batchSize, L.shape[1], L.shape[2]) )
        for i in range(self.valBatches):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            if self.computeFlow:
                bbox[start:end,...] = tracker.forward(D[start:end,...], L[start:end,...], F[start:end,...])
            else:
                bbox[start:end,...] = tracker.forward(D[start:end,...], L[start:end,...], None)
        # Compute IoU
        iou = Tester.getIntOverUnion(L / (self.imgHeight / 2) - 1, bbox)
        # Report to the log
        print 'IoU in validation set:',NP.average(iou)
        return bbox, iou


    def extractFlow(self):
        if self.computeFlow:
            print 'Computing flow of validation data'
            self.valSet['flow'] = VisualAttention.computeFlowFromBatch(self.valSet['data'][:,0:self.seqLength,...])
            print 'Flow done'

