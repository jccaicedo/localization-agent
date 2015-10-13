-- Imports
py = require('fb.python')
require 'nn'
require 'rnn'
require 'cutorch'
require 'cunnx'
-- Add the directory to the PYTHONPATH env variable:
-- export PYTHONPATH=$PYTHONPATH:/home/juan/workspace/localization-agent/tracking
py.exec([=[import VideoSequenceData as vs]=])
vs = py.reval('vs.VideoSequenceData()')

gpu = true

-- ConvNet
net = nn.Sequential()
net:add( nn.Sequencer( nn.SpatialConvolution(2,64,11,11,4,4,2,2) ) )       -- 224 -> 55
net:add( nn.Sequencer( nn.SpatialMaxPooling(3,3,2,2) ) )                   -- 55 ->  27
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(64,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(64,192,5,5,1,1,2,2) ) )       --  27 -> 27
net:add( nn.Sequencer( nn.SpatialMaxPooling(3,3,2,2) ) )                   --  27 ->  13
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(192,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(192,384,3,3,1,1,1,1) ) )      --  13 ->  13
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(384,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(384,256,3,3,1,1,1,1) ) )      --  13 ->  13
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(256,nil,nil,false) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(256,256,3,3,1,1,1,1) ) )      --  13 ->  13
net:add( nn.Sequencer( nn.SpatialMaxPooling(3,3,2,2) ) )                   -- 13 -> 6
net:add( nn.Sequencer( nn.ReLU(true) ) )
--net:add( nn.Sequencer( nn.SpatialBatchNormalization(256,nil,nil,false) ) )

net:add( nn.Sequencer( nn.View(256*6*6) ) )
--net:add( nn.Sequencer( nn.Dropout(0.5) ) )
net:add( nn.Sequencer( nn.Linear(256*6*6, 4096) ) )
--net:add( nn.Sequencer( nn.Threshold(0, 1e-6) ) )
--net:add( nn.Sequencer( nn.Dropout(0.5) ) )
--net:add( nn.Sequencer( nn.Linear(4096, 4096) ) )
--net:add( nn.Sequencer( nn.Threshold(0, 1e-6) ) )
--net:add( nn.Sequencer( nn.Linear(4096, 1000) ) )

-- RNN
inputSize = 4096
hiddenSize = 1024
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

timer = torch.Timer()
t = torch.Timer()
while i < iterations do
   -- a batch of inputs
   vs.prepareSequence()
   local inputs = {}
   local targets = {}
   local j = 1
   while py.eval(vs.nextStep()) do
     I = py.eval(vs.getFrame())
     O = py.eval(vs.getMove())
     if gpu then
       inputs[j] = I:cuda()
       targets[j] = torch.Tensor(O):cuda()
     else
       inputs[j] = I
       targets[j] = torch.Tensor(O)
     end
     j = j + 1
   end
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


