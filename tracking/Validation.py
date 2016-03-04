import numpy as NP
import VisualAttention
import Tester
import h5py

class Validation(object):

    def __init__(self, valBatches, batchSize, generator, imgHeight):
        self.valBatches = valBatches
        self.batchSize = batchSize
        filepath = '_valset_' + str(imgHeight) + '.h5'
        try:
            # Try loading a preexisting validation set
            h5f = h5py.File(filepath, 'r')
            self.valSet = {'data':h5f['data'][:], 'labels':h5f['labels'][:]}
            print 'Loaded validation set with examples:',  self.valSet['data'].shape
            return
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
            print 'Generated validation set with examples:',  self.valSet['data'].shape
            h5f = h5py.File(filepath, "w")
            dset = h5f.create_dataset("data", data=self.valSet['data'])
            lset = h5f.create_dataset("labels", data=self.valSet['labels'])
            h5f.close()
            print 'Validation file saved'

    
    def validate(self, tracker, seqLen=60):
        # Get predictions
        L = self.valSet['labels'][:,0:seqLen,...]
        D = self.valSet['data'][:,0:seqLen,...]
        bbox = NP.zeros( (self.valBatches*self.batchSize, L.shape[1], L.shape[2]) )
        for i in range(self.valBatches):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            bbox[start:end,...] = tracker.forward(D[start:end,...], L[start:end,...])
        # Compute IoU
        iou = Tester.getIntOverUnion(L, bbox)
        # Report to the log
        print 'IoU in validation set:',NP.average(iou)
         
