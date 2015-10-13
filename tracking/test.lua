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
--net:add( nn.Sequencer( nn.Dropout(0.5) ) )
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
lr = 0.0001
updateInterval = 10
iterations = 100000
i = 1
avgErr = 0
maxLength = 10
t = torch.Timer()
simulationFile = '/home/jccaicedo/data/tracking/simulations/simulation.hdf5'

<<<<<<< HEAD
timer = torch.Timer()
t = torch.Timer()
=======
print('Training begins')
>>>>>>> 29ab7e55bcab950a9edbe41cd40574f1d1996091
while i < iterations do
   -- Search simulation data
   while not paths.filep(simulationFile) do
     sys.sleep(0.1)
   end
<<<<<<< HEAD
   local output = net:forward(inputs)
   local err = criterion:forward(output, targets)
   net:zeroGradParameters()
   local gradOutput = criterion:backward(output, targets)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   local netGrad = net:backward(inputs, gradOutput)
   net:updateParameters(lr)

   i = i + 1
   avgErr = avgErr + err
   -- note that updateInterval < rho
   if i % updateInterval == 0 then
      print('Error at iteration ',i,avgErr/updateInterval)
      print('Timing ', t:time().real)
      t = torch.Timer()
      avgErr = 0
=======
   -- Read all sequences in the simulation file
   local data = hdf5.open(simulationFile, 'r')
   local frames = {}
   local moves = {}
   for j=0,99 do
     frames[j] = data:read('frames'..j):all()
     moves[j] = data:read('targets'..j):all()
>>>>>>> 29ab7e55bcab950a9edbe41cd40574f1d1996091
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

-- Do a test prediction
length = math.ceil(math.random()*maxLength)
S = stp.generateMaskedSeq(py.int(length),py.int(32),py.int(32),py.int(6))
stp.showSequence(S[0],S[1],2)
local I = py.eval(S[0])
local T = py.eval(S[1])
local inputs = {}
local targets = {}
for j=1,length do
  inputs[j] = I[{{j},{},{},{}}]
  targets[j] = T[j]
end
local output = net:forward(inputs)
for j=1,length do
  m,i = output[j]:max(1)
  print(targets[j],i[1])
end


