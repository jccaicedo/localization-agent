import h5py
import numpy as np
import time
import VideoSequenceData as vsd
import os,sys
from multiprocessing import Process, JoinableQueue, Queue

inputFile = vsd.dataDir + 'input.hdf5' # Input file for tracker
responseFile = vsd.dataDir + 'output.hdf5' # Output of the tracker
evalSeqDir = '/home/jccaicedoru/data/tracking/simulations/test/'

SEQLEN = 60 # Number of input frames that the tracker takes

# Main Procedure
if __name__ == '__main__':
  
  seq = vsd.VideoSequenceData()
  seq.prepareSequence(loadSequence=evalSeqDir)
  frames = np.zeros((SEQLEN,vsd.channels,vsd.imgSize,vsd.imgSize))
  frames[0,...] = seq.getFrame()
  step = 1
  while seq.nextStep(mode='test'):
    print 'Working at step ',step

    # Read frames and keep up to SEQLEN frames
    if step < SEQLEN:
      frames[step,...] = seq.getFrame()
      outFrames = frames[0:step,...]
    else:
      frames[0:SEQLEN-1,...] = frames[1:SEQLEN,...]
      frames[-1,...] = seq.getFrame()
      outFrames = frames
    step += 1

    # Store the sequence
    output = h5py.File(inputFile,'w')
    output.create_dataset("sequence",data=outFrames)
    output.close()
    os.system('touch ' + inputFile + '.ready')

    # Wait a bit more while the tracker puts the answer
    while not os.path.exists(responseFile) or not os.path.exists(responseFile + '.ready'):
      time.sleep(0.1)

    # Read the response of the tracker
    response = h5py.File(responseFile)
    data = np.asarray(response['predictions'][...])
    response.close()
    os.remove(responseFile)
    os.remove(responseFile+'.ready')

    # Update the sequencer
    seq.setMove(data)

