import argparse as AP
import RecurrentTracker
import pickle

from CaffeCnn import CaffeCnn
from TheanoGruRnn import TheanoGruRnn
from GaussianGenerator import GaussianGenerator

class Controller(object):
    
    def train(self, tracker, epochs, batches, batchSize, generator, trackerModelPath, imgHeigh):
        for i in range(0, epochs):
            train_cost = test_cost = 0
            for j in range(0, batches):
                data, label = generator.getBatch(batchSize)
        
                print 'Initial data shape:', data.shape
                if generator.grayscale:
                    data = data[:, :, NP.newaxis, :, :]
                data /= 255.0
                print 'Final data shape:', data.shape
                label = label / (imgHeigh / 2.) - 1.
        
                cost, bbox_seq = tracker.fit(data, label)
                
                print i, j, cost
                train_cost += cost
            print 'Epoch average loss (train, test)', train_cost / 2000, test_cost / 2000
            
            with open(trackerModelPath + str(i), 'wb') as trackerFile:
                pickle.dump(tracker.rnn.params, trackerFile, pickle.HIGHEST_PROTOCOL)
                
### Utility functions

def build_parser():
    parser = AP.ArgumentParser(description='Trains a RNN tracker')
    parser.add_argument('--dataDir', help='Directory of trajectory model', type=str, default='/home/fmpaezri/repos/localization-agent/notebooks')
    parser.add_argument('--epochs', help='Number of epochs for training', type=int, default=4)
    parser.add_argument('--batches', help='Number of batches per epoch', type=int, default=2000)
    parser.add_argument('--batchSize', help='Number of elements in batch', type=int, default=4)
    parser.add_argument('--gruInputDim', help='Number of input features ', type=int, default=50180)
    parser.add_argument('--imgHeigh', help='Imgage Heigh', type=int, default=224)
    parser.add_argument('--imgWidth', help='Image width', type=int, default=224)
    parser.add_argument('--gruStateDim', help='Dimension of GRU state', type=int, default=256)
    parser.add_argument('--seqLength', help='Length of sequences', type=int, default=60)
    parser.add_argument('--trackerModelPath', help='Name of model file', type=str, default='model.pkl')
    parser.add_argument('--cnnModelPath', help='Name of model file', type=str)
    parser.add_argument('--zeroTailFc', help='', type=bool, default=False)
    return parser

if __name__ == '__main__':
    
    # Configuration
    
    parser = build_parser()
    args = parser.parse_args()
    globals().update(vars(args))
    
    
    cnn = CaffeCnn(imgHeigh, imgWidth, deployPath, cnnModelPath, caffeRoot, batchSize, seqLength)
    rnn = TheanoGruRnn(gruInputDim, gruStateDim, batchSize, seqLength, zeroTailFc)
    
    if (modelName != ""):
        rnn.loadModel(modelName)
    
    tracker = RecurrentTracker(cnn, rnn)
    
    generator = GaussianGenerator(dataDir=dataDir, seqLength=seq_len, imageSize=img_row, grayscale=False)
    
    controller = Controller()
    controller.train(tracker, epochs, batches, batchSize, generator, cnnModelPath, imgHeigh)
    
