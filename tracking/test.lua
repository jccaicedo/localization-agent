-- Imports
py = require('fb.python')
require 'nn'
require 'rnn'
-- Add the directory to the PYTHONPATH env variable:
-- export PYTHONPATH=$PYTHONPATH:/home/juan/workspace/localization-agent/tracking
py.exec([=[import TrajectorySimulator]=])
ts = py.reval('TrajectorySimulator')

-- ConvNet
net = nn.Sequential()
net:add( nn.Sequencer( nn.SpatialConvolution(2,16,5,5,1,1) ) )
net:add( nn.Sequencer( nn.ReLU() ) )
net:add( nn.Sequencer( nn.SpatialMaxPooling(2,2,2,2) ) )
net:add( nn.Sequencer( nn.SpatialConvolution(16,6,3,3,1,1) ) )
net:add( nn.Sequencer( nn.ReLU() ) )
net:add( nn.Sequencer( nn.SpatialMaxPooling(2,2,2,2) ) )
net:add( nn.Sequencer( nn.View(6*6*6) ) )
net:add( nn.Sequencer( nn.Linear(6*6*6,100) ) )
-- net:add( nn.Sequencer( nn.Dropout(0.5) ) )

-- RNN
inputSize = 100
hiddenSize = 50
nIndex = 5

-- LSTM
lstm = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))

-- Prediction Layers
net:add( lstm )
net:add( nn.Sequencer( nn.Linear(hiddenSize,nIndex) ) )
net:add( nn.Sequencer( nn.LogSoftMax() ) )
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- Training
lr = 0.01
updateInterval = 100
iterations = 5000
i = 1
avgErr = 0
maxLength = 10
scene = '/home/juan/Pictures/test/bogota.jpg'
object = '/home/juan/Pictures/test/photo.jpg'
polygon = {50, 0, 100, 50, 50, 100, 0, 50}

while i < iterations do
   -- a batch of inputs
   S = ts.TrajectorySimulator(scene, object, polygon) 
   local inputs = {}
   local targets = {}
   while py.eval(S.nextStep()) do
     inputs[j] = py.eval(S.getMaskedFrame())
     targets[j] = py.eval(S.getMove())
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
      print(i,avgErr/updateInterval)
      avgErr = 0
   end
end

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


