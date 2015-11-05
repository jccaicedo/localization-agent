-- Imports
require 'nn'
require 'rnn'
require 'cutorch'
require 'cunnx'

require 'sys'
require 'paths'
require 'hdf5'

inputFile = arg[1]
outputFile = arg[2]

gpu = true
workingDir = '/data1/vot-challenge/simulations/'
workingDir = '/home/jccaicedo/data/tracking/simulations/test/'

net = torch.load(workingDir .. 'net.snapshot.bin')

-- GPU based
if gpu then
  net = net:cuda()
end

dataFile = workingDir .. inputFile
predictionsFile = workingDir .. outputFile

timer = torch.Timer()
t = torch.Timer()
globalTimer = torch.Timer()
timeout = 5
net:evaluate()
local i = 1
local keepRunning = paths.filep(dataFile .. '.running') and (globalTimer:time().real < timeout)
print('Test begins', net)
while keepRunning do
   -- Search input data
   while (not paths.filep(dataFile) or not paths.filep(dataFile .. '.ready')) and keepRunning do
     sys.sleep(0.1)
     keepRunning = paths.filep(dataFile .. '.running') and (globalTimer:time().real < timeout)
   end
   if not keepRunning then
     break
   end
   print('Waiting file '..t:time().real)
   t = torch.Timer()
   -- Read the sequence in the input file
   local data = hdf5.open(dataFile, 'r')
   local frames = data:read('sequence'):all()
   data:close()
   sys.execute('rm ' .. dataFile)
   sys.execute('rm ' .. dataFile .. '.ready')
   print('Reading file '..t:time().real)
   t = torch.Timer()
   if gpu then
     frames = frames:cuda()
   end
   -- Prepare tables for the sequence
   local inputs = {}
   local s = frames.size(frames)
   for k=1,s[1] do
     inputs[k] = frames[{ {k}, {}, {}, {}  }]
   end
   print('Preparing data '..t:time().real)
   t = torch.Timer()
   -- Feed data to the network
   net:forget()
   local output = net:forward(inputs)
   outFile = hdf5.open(predictionsFile, 'w')
   outFile:write('predictions', output[#output]:float())
   outFile:close()
   sys.execute('touch ' .. predictionsFile .. '.ready')

   --print('Frame '..i,t:time().real)
   t = torch.Timer()
   globalTimer = torch.Timer()
   i = i + 1
   keepRunning = paths.filep(dataFile .. '.running') and (globalTimer:time().real < timeout)
end
--print('Total tracking time: ' .. timer:time().real)

