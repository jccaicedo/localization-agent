print 'Starting'
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

from ObjectLocalizerEnvironment import ObjectLocalizerEnvironment
from DeepQNetwork import DeepQNetwork
from DeepQLearning import DeepQLearning
from MDPObjectLocalizerTask import MDPObjectLocalizerTask
from ObjectLocalizationAgent import ObjectLocalizationAgent

imgsDir = '/u/sciteam/caicedor/scratch/pascalImgs'
candidatesFile = '/u/sciteam/caicedor/cnnPatches/trainingDetections/aeroplane_0.001_9ScalePatchesNoDups_0.5_2.out.result.log'
groundTruth = '/u/sciteam/caicedor/cnnPatches/lists/2007/trainval/aeroplane_gt_bboxes.txt'

print 'Starting Environment'
environment = ObjectLocalizerEnvironment(imgsDir, candidatesFile, 'Training')
print 'Initializing DeepQNetwork'
controller = DeepQNetwork()
print 'Initializing Q Learner'
learner = DeepQLearning()
print 'Preparing Agent'
agent = ObjectLocalizationAgent(controller, learner)
print 'Configuring Task'
task = MDPObjectLocalizerTask(environment, groundTruth)
print 'Setting up Experiment'
experiment = Experiment(task, agent)
i = 0
print 'Main Loop'
while True:
  experiment.doInteractions(100)
  agent.learn()
  agent.reset()
  print 'Epoch',i
  i += 1

