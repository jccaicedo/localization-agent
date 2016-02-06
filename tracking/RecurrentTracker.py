class RecurrentTracker(object):
    cnn = None
    rnn = None
    
    
    def __init__(self, cnn, rnn):
        self.cnn = cnn
        self.rnn = rnn
        
        
    def fit(self, data, label):
        out, activations = self.cnn.forward(data)
        cost, bbox_seq = self.rnn.fit(activations, label)
        
        return cost, bbox_seq
            
    
    def forward(self, data, label):
        out, activations = self.cnn.forward(data)
        pred = self.rnn.forward(activations, label)
        
        return pred