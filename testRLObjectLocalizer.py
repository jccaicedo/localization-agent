import os, sys
import RLConfig as config
if len(sys.argv) < 2:
  print 'Use: trainRLObjectLocalizer.py configFile'
  sys.exit()

config.readConfiguration(sys.argv[1])

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

from ObjectLocalizerEnvironment import ObjectLocalizerEnvironment
from DeepQNetwork import DeepQNetwork
from DeepQLearning import DeepQLearning
from MDPObjectLocalizerTask import MDPObjectLocalizerTask
from ObjectLocalizationAgent import ObjectLocalizationAgent


print 'Starting Environment'
environment = ObjectLocalizerEnvironment(config.get('imageDir'), config.get('testFile'), 'Testing')
print 'Initializing DeepQNetwork'
controller = DeepQNetwork()
#print 'Initializing Q Learner'
#learner = DeepQLearning()
print 'Preparing Agent'
agent = ObjectLocalizationAgent(controller)
print 'Configuring Task'
task = MDPObjectLocalizerTask(environment, config.get('groundTruth'))
print 'Setting up Experiment'
experiment = Experiment(task, agent)
i = 0
print 'Main Loop'
while i < config.geti('maximumEpochs'):
  experiment.doInteractions(int(config.get('numInteractions')))
  agent.learn()
  agent.reset()
  print 'Epoch',i
  i += 1

