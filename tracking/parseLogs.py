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
    valIoU = [float(l.split()[-1]) for l in log if l.startswith('IoU')]
    
    # Compute basic statistics
    totalBatches = len(cost)
    totalEpochs = len(avgEpoch)
    batchesPerEpoch = totalBatches/totalEpochs 
    avgs = [ (x*seqPerEpoch)/batchesPerEpoch for x in avgEpoch]
    
    maxCost = min(np.max(cost), 1.0)
    minCost = np.min(cost)
    maxIdx = np.argmax(cost)
    minIdx = np.argmin(cost)

    # Prepare plot
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,10))

    axes = plt.gca()
    axes.set_ylim([0., maxCost])

    # Compute time info
    epochPos = (np.arange(totalEpochs)+1)*batchesPerEpoch-batchesPerEpoch/2
    totalTime = 0
    for i in range(len(avgs)):
        axes.text(epochPos[i]*0.95, avgs[i]*1.2, '%5.0fm'%(timeEpoch[i]/60), fontsize=12, weight='bold')
        totalTime += timeEpoch[i]
    totalTime /= (60*60)
    plt.title('Training cost. (Experiment Time: %3.2f hours)'%totalTime)
 
    # Create annotations    
    axes.annotate('Max Cost: %1.3f'%(np.max(cost)), xy=(maxIdx, maxCost), xycoords='data',
                    xytext=(50, -50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->") )
    axes.annotate('Min Cost: %1.3f'%(minCost), xy=(minIdx, minCost), xycoords='data',
                    xytext=(-150, 150), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->") )
    axes.annotate('Max IoU: %1.3f'%(np.max(valIoU)), xy=(epochPos[np.argmax(valIoU)], np.max(valIoU)), xycoords='data',
                    xytext=(-150, 50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->") )
    axes.annotate('Min IoU: %1.3f'%(np.min(valIoU)), xy=(epochPos[np.argmin(valIoU)], np.min(valIoU)), xycoords='data',
                    xytext=(50, -50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->") )
    axes.set_ylabel('Cost')
    axes.set_xlabel('Minibatches')
    axes.text(len(cost)/2, maxCost/1.2, 
            'Epochs=%d\nbatchSize=%d\nlRate=%f\nGRUsize=%d'%(totalEpochs,batchSize,lrate,gruSize), 
            style='italic', bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
    
   
    # Make plot
    plt.plot(cost, label='Minibatches')
    plt.plot(epochPos, avgs, 'r-o', markersize=20, linewidth=2.0, label='Epoch Average')
    plt.plot(epochPos, valIoU, 'y-*', markersize=20, linewidth=1.0, label='Validation IoU')
    plt.legend()
    plt.legend(bbox_to_anchor=(0., 1.05, 1., 0.102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='16')
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

    seqPerEpoch = 9600
    parseAndPlot(log_file, out_file, batch_size, learning_rate, gru_dim, seqPerEpoch)

