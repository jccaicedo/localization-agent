import numpy as NP
import time

# TODO: parameterize this value that sets the size of the replay memory
MEMORY_FACTOR = 20 # Number of minibatches to keep

class RecurrentTracker(object):
    cnn = None
    rnn = None
    memory = None
    
    
    def __init__(self, cnn, rnn):
        self.cnn = cnn
        self.rnn = rnn
        
        
    def fit(self, data, label, store_in_mem):
        st = time.time()
        activations = self.cnn.forward(data) if self.cnn is not None else data
        print 'Forwarding: {}'.format(time.time()-st)
        cost, bbox_seq = self.rnn.fit(activations, label)
        if store_in_mem:
            self.store(activations, label)
        
        return cost, bbox_seq
            
    
    def forward(self, data, label):
        activations = self.cnn.forward(data) if self.cnn is not None else data
        pred = self.rnn.forward(activations, label)
        
        return pred

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

