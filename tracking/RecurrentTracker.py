import numpy as NP
import time
import logging

# TODO: parameterize this value that sets the size of the replay memory
MEMORY_FACTOR = 50 # Number of minibatches to keep

class RecurrentTracker(object):
    cnn = None
    rnn = None
    memory = None
    
    
    def __init__(self, cnn, rnn):
        self.cnn = cnn
        self.rnn = rnn
        
    def activate(self, data):
        if self.cnn is not None:
            activations = self.cnn.forward(data)
        else:
            activations = data
        return activations
        
    def fit(self, data, label, flow, store_in_mem):
        st = time.time()
        activations = self.activate(data)
        cost, bbox_seq, transformed = self.processMultiBatch(activations, label, flow, self.rnn.batchSize, (label.shape[-1],), self.rnn.fit)
        if store_in_mem:
            self.store(activations, label)
        
        return cost, bbox_seq, transformed

    def decayLearningRate(self):
        self.rnn.decayLearningRate()
        logging.info('Current learning rate: {}'.format(self.rnn.learningRate.get_value(borrow=True),))
    
    def forward(self, data, label, flow):
        activations = self.activate(data)
        cost, pred, transformed = self.processMultiBatch(activations, label, flow, self.rnn.batchSize, (label.shape[-1],), self.rnn.forward)
        
        return pred, transformed

    def processMultiBatch(self, data, label, flow, atomicBatchSize, outputShape, processFunction):
        '''Reshapes input to fit batchSize and applies the specified function over each atomic batch'''
        #Initialize output variables
        outputs = NP.zeros((data.shape[0], data.shape[1])+outputShape, dtype=data.dtype)
        crops = NP.zeros(data.shape, dtype=data.dtype)

        totalCost = 0
        #Reshape inputs, adding a new dim
        if data.shape[0] != label.shape[0]:
            raise Exception('Data and labels first shape must match: {} != {}'.format(data.shape, label.shape))
        newDataShape = (data.shape[0]/atomicBatchSize, atomicBatchSize) + data.shape[-4:]
        logging.debug('Reshaping data from %s to %s', data.shape, newDataShape)
        newLabelShape = (label.shape[0]/atomicBatchSize, atomicBatchSize) + label.shape[-2:]
        logging.debug('Reshaping labels from %s to %s', label.shape, newLabelShape)
        data = data.reshape(newDataShape)
        label = label.reshape(newLabelShape)
        if flow is not None:
            newFlowShape = (flow.shape[0]/atomicBatchSize, atomicBatchSize) + flow.shape[-4:]
            logging.debug('Reshaping flow from %s to %s', flow.shape, newFlowShape)
            flow = flow.reshape(newFlowShape)
        #Iterate and accumulate
        for i in NP.arange(newDataShape[0]):
            if flow is not None:
                cost, bbox_seq, transformed = processFunction(data[i], label[i], flow[i])
            else:
                cost, bbox_seq, transformed = processFunction(data[i], label[i], flow)
            totalCost += cost
            outputs[i*atomicBatchSize:(i+1)*atomicBatchSize, ...] = bbox_seq
            crops[i*atomicBatchSize:(i+1)*atomicBatchSize, ...] = transformed
        return totalCost, outputs, crops

    def store(self, activations, labels):
        if self.memory is None:
            self.memory = {}
            capacity = MEMORY_FACTOR*activations.shape[0] # Store 100 precomputed batches
            self.memory['A'] = NP.zeros([capacity] + list(activations.shape[1:]))
            self.memory['L']= NP.zeros([capacity] + list(labels.shape[1:]))
            self.memory['p']= 0
            self.memory['full'] = False

        if self.memory['p'] < self.memory['A'].shape[0]:
            start = self.memory['p']
            end = start + activations.shape[0]
            self.memory['A'][start:end,...] = activations
            self.memory['L'][start:end,...] = labels
            self.memory['p'] += activations.shape[0]

        if self.memory['p'] >= self.memory['A'].shape[0]:
            self.memory['p'] = 0
            self.memory['full'] = True

    def sampleFromMem(self):
        if self.memory is None:
            return False
        elif not self.memory['full']:
            return False
        elif NP.random.rand() <= 0.5:
            return False
        else:
            return True

    def getSample(self, batchSize):
        idx = NP.random.permutation( self.memory['A'].shape[0] )[0:batchSize]
        return (self.memory['A'][idx,...], self.memory['L'][idx,...])

