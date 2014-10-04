import os, sys
import RLConfig as config
if len(sys.argv) < 5:
  print 'Use: trainRLObjectLocalizer.py configFile detectionsFile groundTruths outputFile'
  sys.exit()

config.readConfiguration(sys.argv[1])
detectionsFile = sys.argv[2]
groundTruth = sys.argv[3]
outputFile = sys.argv[4]

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

from ObjectLocalizerEnvironment import ObjectLocalizerEnvironment
from DeepQNetwork import DeepQNetwork
from DeepQLearning import DeepQLearning
from MDPObjectLocalizerTask import MDPObjectLocalizerTask
from ObjectLocalizationAgent import ObjectLocalizationAgent

print 'Starting Environment'
environment = ObjectLocalizerEnvironment(config.get('imageDir'), detectionsFile, 'Testing')
print 'Initializing DeepQNetwork'
controller = DeepQNetwork()
print 'Preparing Agent'
agent = ObjectLocalizationAgent(controller)
print 'Configuring Task'
task = MDPObjectLocalizerTask(environment, groundTruth)
print 'Setting up Experiment'
experiment = Experiment(task, agent)
print 'Main Loop'
while environment.hasMoreEpisodes():
  experiment.doInteractions(1)
print 'All test episodes done'
environment.saveRecords(outputFile)
