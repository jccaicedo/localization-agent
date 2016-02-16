class RecurrentTracker(object):
    cnn = None
    rnn = None
    
    
    def __init__(self, cnn, rnn):
        self.cnn = cnn
        self.rnn = rnn
        
        
    def fit(self, data, label):
        activations = self.cnn.forward(data) if self.cnn is not None else data
        cost, bbox_seq = self.rnn.fit(activations, label)
        
        return cost, bbox_seq
            
    
    def forward(self, data, label):
        activations = self.cnn.forward(data) if self.cnn is not None else data
        pred = self.rnn.forward(activations, label)
        
        return pred