import h5py
import numpy as np
import time
import VideoSequenceData as vsd
import os,sys
from multiprocessing import Process, JoinableQueue, Queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

testDir = '/home/jccaicedo/data/tracking/simulations/test/'
inputFile = testDir + 'input.hdf5' # Input file for tracker
responseFile = testDir + 'output.hdf5' # Output of the tracker
loadFrom = testDir
#loadFrom = 'TraxClient'

SEQLEN = 6 # Number of input frames that the tracker takes

def writeRequest(packet):
  output = h5py.File(inputFile,'w')
  output.create_dataset("sequence",data=packet)
  output.close()
  os.system('touch ' + inputFile + '.ready')

def readResponse():
  response = h5py.File(responseFile)
  data = np.asarray(response['predictions'][...])
  response.close()
  os.remove(responseFile)
  os.remove(responseFile+'.ready')
  return data

def saveSequenceImages(seq,step,dir):
  s = seq.shape[0]
  for i in range(s):
    plt.imshow(np.swapaxes(np.swapaxes(seq[i,[3,2,0],:,:],0,2),0,1))
    plt.savefig(dir + 'seq_' + str(step) + '_' + str(i) + '.png')

# Main Procedure
if __name__ == '__main__':
  
  torchProcess = False
  seq = vsd.VideoSequenceData()
  seq.prepareSequence(loadSequence=loadFrom)
  frames = np.zeros((SEQLEN,vsd.channels,vsd.imgSize,vsd.imgSize))
  frames[0,...] = seq.getFrame()
  step = 1
  os.system('touch ' + inputFile + '.running')
  predictions = []
  while seq.nextStep(mode='test'):
    print 'Working at step ',step

    # Read frames and keep up to SEQLEN frames
    if step < SEQLEN:
      frames[step,...] = seq.getFrame()
      outFrames = frames[0:step,...]
    else:
      frames[0:SEQLEN-1,...] = frames[1:SEQLEN,...]
      frames[-1,...] = seq.getFrame()
      mask = vsd.makeMask(vsd.imgSize,vsd.imgSize,predictions[step-SEQLEN])
      frames[:,-1,:,:] = mask
      outFrames = frames
    #saveSequenceImages(outFrames,step,testDir)
    step += 1

    # Store the sequence
    writeRequest(outFrames)

    # Start the torch process if it is not already running
    if not torchProcess:
      os.system('th rnns/test_CNN_RNN.lua &')
      torchProcess = True

    # Wait a bit more while the tracker puts the answer
    while not os.path.exists(responseFile) or not os.path.exists(responseFile + '.ready'):
      time.sleep(0.1)

    # Read the response of the tracker
    data = readResponse()

    # Update the sequencer
    seq.setMove(data)
    predictions.append( seq.predictedBox[:] )

os.remove(inputFile + '.running')
print 'Finished'
