-- Imports
require 'nn'
require 'rnn'
require 'cutorch'
require 'cunnx'

require 'sys'
require 'paths'
require 'hdf5'

gpu = true

net = torch.load('/home/jccaicedoru/data/tracking/simulations/net.snapshot.bin')

-- GPU based
if gpu then
  net = net:cuda()
end

seqFrames = 80
dataFile = '/home/jccaicedoru/data/tracking/simulations/input.hdf5'
predictionsFile = '/home/jccaicedoru/data/tracking/simulations/output.hdf5'

timer = torch.Timer()
t = torch.Timer()
net:evaluate()
print('Test begins')
for i = 1,seqFrames do
   -- Search input data
   while not paths.filep(dataFile) or not paths.filep(dataFile .. '.ready') do
     sys.sleep(0.1)
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
end
print('Total tracking time: ' .. timer:time().real)

