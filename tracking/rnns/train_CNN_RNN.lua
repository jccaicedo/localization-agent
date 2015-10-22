-- Imports
require 'nn'
require 'rnn'
require 'cutorch'
require 'cunnx'

-- Add the directory to the PYTHONPATH env variable:
-- export PYTHONPATH=$PYTHONPATH:/home/juan/workspace/localization-agent/tracking
--py = require('fb.python')
--py.exec([=[import VideoSequenceData as vs]=])
--vs = py.reval('vs.VideoSequenceData()')

require 'sys'
require 'paths'
require 'hdf5'

gpu = true
workingDir = '/home/jccaicedoru/data/tracking/simulations/'

-- ConvNet
net = nn.Sequential()
net:add( nn.Sequencer( nn.SpatialConvolution(4,64,5,5,2,2,1,1) ) )       --  64 -> 32
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(192,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(64,128,3,3,1,1,2,2) ) )      --  32 -> 32
net:add( nn.Sequencer( nn.SpatialMaxPooling(3,3,2,2) ) )                   -- 32 -> 16
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(384,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(128,128,3,3,1,1,1,1) ) )      --  16 ->  15
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(256,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(128,128,3,3,1,1,1,1) ) )      --  15 -> 14
net:add( nn.Sequencer( nn.SpatialMaxPooling(3,3,2,2) ) )                   -- 14 -> 7
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(256,nil,nil,false) ) )

net:add( nn.Sequencer( nn.View(128*7*7) ) )
net:add( nn.Sequencer( nn.Dropout(0.5) ) )
net:add( nn.Sequencer( nn.Linear(128*7*7, 2048) ) )
--net:add( nn.Sequencer( nn.Threshold(0, 1e-6) ) )
--net:add( nn.Sequencer( nn.Dropout(0.5) ) )
--net:add( nn.Sequencer( nn.Linear(4096, 4096) ) )
--net:add( nn.Sequencer( nn.Threshold(0, 1e-6) ) )
--net:add( nn.Sequencer( nn.Linear(4096, 1000) ) )

-- RNN
inputSize = 2048
hiddenSize = 512
nIndex = 4

-- LSTM
lstm = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))

-- Prediction Layers
net:add( lstm )
net:add( nn.Sequencer( nn.Linear(hiddenSize,nIndex) ) )
--net:add( nn.Sequencer( nn.LogSoftMax() ) )
criterion = nn.SequencerCriterion(nn.MSECriterion())

-- GPU based
if gpu then
  net = net:cuda()
  criterion = criterion:cuda()
end

-- Training
lr = 0.001
updateInterval = 10
iterations = 100000
i = 1
avgErr = 0
t = torch.Timer()
simulationFile = workingDir .. 'simulation.hdf5'

timer = torch.Timer()
t = torch.Timer()
net:training()
print('Training begins')
while i < iterations do
   -- Search simulation data
   while not paths.filep(simulationFile) do
     sys.sleep(0.1)
   end
   -- Read all sequences in the simulation file
   local data = hdf5.open(simulationFile, 'r')
   local frames = {}
   local moves = {}
   for j=0,99 do
     frames[j] = data:read('frames'..j):all()
     moves[j] = data:read('targets'..j):all()
   end
   data:close()
   sys.execute('rm ' .. simulationFile)
   -- Use all read sequences
   for j=0,99 do
     if gpu then
       frames_j = frames[j]:cuda()
       moves_j = moves[j]:cuda()
     end
     -- Prepare tables for one sequence
     local inputs = {}
     local targets = {}
     for k=1,60 do
       inputs[k] = frames_j[{ {k}, {}, {}, {}  }]
       targets[k] = moves_j[{ {k}, {} }]
     end
     -- Feed data to the network
     local output = net:forward(inputs)
     local err = criterion:forward(output, targets)
     net:zeroGradParameters()
     -- Update network parameters
     local gradOutput = criterion:backward(output, targets)
     local netGrad = net:backward(inputs, gradOutput)
     net:updateParameters(lr)
     -- Update counters and print messages
     i = i + 1
     avgErr = avgErr + err
     if i % updateInterval == 0 then
        print(i,avgErr/updateInterval,t:time().real)
        avgErr = 0
        t = torch.Timer()
     end
  end
end
print('Total training time: ' .. timer:time().real)
torch.save('net.snapshot.bin', net)

