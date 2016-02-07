import time

class RecurrentTracker(object):
    cnn = None
    rnn = None
    
    
    def __init__(self, cnn, rnn):
        self.cnn = cnn
        self.rnn = rnn
        
        
    def fit(self, data, label):
        st = time.time()
        activations = self.cnn.forward(data) if self.cnn is not None else data
        print 'Forwarding: {}'.format(time.time()-st)
        cost, bbox_seq = self.rnn.fit(activations, label)
        
        return cost, bbox_seq
            
    
    def forward(self, data, label):
        activations = self.cnn.forward(data) if self.cnn is not None else data
        pred = self.rnn.forward(activations, label)
        
        return pred
