import h5py
import numpy as np
import time
import VideoSequenceData as vsd
import os,sys
from multiprocessing import Process, JoinableQueue, Queue

inputFile = vsd.dataDir + 'input.hdf5' # Input file for tracker
responseFile = vsd.dataDir + 'output.hdf5' # Output of the tracker
loadFrom = vsd.dataDir + 'test/'
#loadFrom = 'TraxClient'

SEQLEN = 60 # Number of input frames that the tracker takes

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

# Main Procedure
if __name__ == '__main__':
  
  torchProcess = False
  seq = vsd.VideoSequenceData()
  seq.prepareSequence(loadSequence=loadFrom)
  frames = np.zeros((SEQLEN,vsd.channels,vsd.imgSize,vsd.imgSize))
  frames[0,...] = seq.getFrame()
  step = 1
  os.system('touch ' + inputFile + '.running')
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

os.remove(inputFile + '.running')
print 'Finished'
