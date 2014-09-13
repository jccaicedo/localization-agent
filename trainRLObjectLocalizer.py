print 'Starting'
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

from ObjectLocalizerEnvironment import ObjectLocalizerEnvironment
from DeepQNetwork import DeepQNetwork
from DeepQLearning import DeepQLearning
from MDPObjectLocalizerTask import MDPObjectLocalizerTask
from ObjectLocalizationAgent import ObjectLocalizationAgent

import RLConfig as config

print 'Starting Environment'
environment = ObjectLocalizerEnvironment(config.imageDir, config.candidatesFile, 'Training')
print 'Initializing DeepQNetwork'
controller = DeepQNetwork()
print 'Initializing Q Learner'
learner = DeepQLearning()
print 'Preparing Agent'
agent = ObjectLocalizationAgent(controller, learner)
print 'Configuring Task'
task = MDPObjectLocalizerTask(environment, config.groundTruth)
print 'Setting up Experiment'
experiment = Experiment(task, agent)
i = 0
print 'Main Loop'
while i < config.maximumEpochs:
  experiment.doInteractions(config.numInteractions)
  agent.learn()
  agent.reset()
  print 'Epoch',i
  i += 1

