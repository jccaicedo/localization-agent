import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap


def parseAndPlot(logFile, outFile, batchSize, lrate, gruSize, seqPerEpoch):
    # Open and parse log file
    log = [l for l in open(logFile, 'r')]
    cost = [float(l.split()[-1]) for l in log if l.startswith('Cost')]
    avgEpoch = [float(l.split()[-1]) for l in log if l.startswith('Epoch average')]
    timeEpoch = [float(l.split()[-1]) for l in log if l.startswith('Epoch time')]
    
    # Compute basic statistics
    totalBatches = len(cost)
    totalEpochs = len(avgEpoch)
    batchesPerEpoch = totalBatches/totalEpochs 
    avgs = [ (x*seqPerEpoch)/batchesPerEpoch for x in avgEpoch]
    
    maxCost = np.max(cost)
    minCost = np.min(cost)
    maxIdx = np.argmax(cost)
    minIdx = np.argmin(cost)
    
    # Prepare plot
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,10))

    axes = plt.gca()
    axes.set_ylim([min(cost),max(cost)])
    axes.annotate('Max Cost: %1.3f'%(maxCost), xy=(maxIdx, maxCost), xycoords='data',
                    xytext=(50, -50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->") )
    axes.annotate('Min Cost: %1.3f'%(minCost), xy=(minIdx, minCost), xycoords='data',
                    xytext=(-150, 150), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->") )
    axes.set_ylabel('Cost')
    axes.set_xlabel('Minibatches')
    axes.text(len(cost)/2, maxCost/1.2, 
            'Epochs=%d\nbatchSize=%d\nlRate=%f\nGRUsize=%d'%(totalEpochs,batchSize,lrate,gruSize), 
            style='italic', bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
    
    # Compute time info
    epochPos = (np.arange(totalEpochs)+1)*batchesPerEpoch-batchesPerEpoch/2
    totalTime = 0
    for i in range(len(avgs)):
        axes.text(epochPos[i]*0.95, avgs[i]*1.2, '%5.0f sec'%(timeEpoch[i]), fontsize=15, weight='bold')
        totalTime += timeEpoch[i]
    totalTime /= (60*60)
    plt.title('Training cost. (Total Time: %3.2f hours)'%totalTime)
    
    # Make plot
    plt.plot(cost, label='Minibatches')
    plt.plot(epochPos, avgs, 'r-o', markersize=20, linewidth=2.0, label='Epoch Average')
    plt.legend()
    plt.savefig(outFile)

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Plots training results')
    parser.add_argument('--log_file', help='File to parse', type=str)
    parser.add_argument('--out_file', help='File to save results', type=str, default='results.png')
    parser.add_argument('--batch_size', help='Number of elements in batch', type=int, default=32)
    parser.add_argument('--gru_dim', help='Dimension of GRU state', type=int, default=256)
    parser.add_argument('--learning_rate', help='SGD learning rate', type=float, default=0.0005)

    args = parser.parse_args()
    globals().update(vars(args))

    seqPerEpoch = 32000
    parseAndPlot(log_file, out_file, batch_size, learning_rate, gru_dim, seqPerEpoch)

