import h5py
import numpy as np
import VideoSequenceData as vsd

# TODO: This script could save thousands of simulations at once

# Prepare a video sequence 
seq = vsd.VideoSequenceData()
seq.prepareSequence()

# Store in a numpy array
simData = np.zeros((60,2,64,64))
step = 0
while seq.nextStep():
  simData[step,:,:,:] = seq.getFrame()
  step += 1

# Send to an HDF5 file
outFile = h5py.File('myfile.hdf5','w')
outFile.create_dataset("sdata",data=simData)
outFile.close()

'''
-- Torch Side
require 'hdf5'
local myFile = hdf5.open('myfile.hdf5', 'r')
local data = myFile:read('sdata'):all()
myFile:close()
'''
