-- Imports
require 'nn'
require 'rnn'
require 'cutorch'
require 'cunnx'

require 'sys'
require 'paths'
require 'hdf5'

gpu = true
workingDir = '/data1/vot-challenge/simulations/'
workingDir = '/home/jccaicedo/data/tracking/simulations/test/'

net = torch.load(workingDir .. 'net.snapshot.bin')

-- GPU based
if gpu then
  net = net:cuda()
end

dataFile = workingDir .. 'input.hdf5'
predictionsFile = workingDir .. 'output.hdf5'

timer = torch.Timer()
t = torch.Timer()
net:evaluate()
local i = 1
local keepRunning = paths.filep(dataFile .. '.running')
print('Test begins')
while keepRunning do
   -- Search input data
   while (not paths.filep(dataFile) or not paths.filep(dataFile .. '.ready')) and keepRunning do
     sys.sleep(0.1)
     keepRunning = paths.filep(dataFile .. '.running')
   end
   if not keepRunning then
     break
   end
   -- Read the sequence in the input file
   local data = hdf5.open(dataFile, 'r')
   local frames = data:read('sequence'):all()
   data:close()
   sys.execute('rm ' .. dataFile)
   sys.execute('rm ' .. dataFile .. '.ready')
   if gpu then
     frames = frames:cuda()
   end
   -- Prepare tables for the sequence
   local inputs = {}
   local s = frames.size(frames)
   for k=1,s[1] do
     inputs[k] = frames[{ {k}, {}, {}, {}  }]
   end
   -- Feed data to the network
   local output = net:forward(inputs)
   outFile = hdf5.open(predictionsFile, 'w')
   outFile:write('predictions', output[#output]:float())
   outFile:close()
   sys.execute('touch ' .. predictionsFile .. '.ready')

   print('Frame '..i,t:time().real)
   t = torch.Timer()
   i = i + 1
   keepRunning = paths.filep(dataFile .. '.running')
end
print('Total tracking time: ' .. timer:time().real)

