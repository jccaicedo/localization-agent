import os, sys
import RLConfig as config
if len(sys.argv) < 2:
  print 'Use: trainRLObjectLocalizer.py configFile'
  sys.exit()

def adjustEpsilon(totalEpochs, currentEpoch, epsilon):
  maxAnnealingEpoch = totalEpochs*0.2
  if currentEpoch > maxAnnealingEpoch or epsilon <= 0.1:
    return 0.1
  else:
    return epsilon - 1/maxAnnealingEpoch

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
epsilon = 1.0
environment = ObjectLocalizerEnvironment(config.get('imageDir'), config.get('candidatesFile'), 'Training')
print 'Initializing DeepQNetwork'
controller = DeepQNetwork()
controller.setEpsilonGreedy(epsilon)
print 'Initializing Q Learner'
learner = DeepQLearning()
print 'Preparing Agent'
agent = ObjectLocalizationAgent(controller, learner)
print 'Configuring Task'
task = MDPObjectLocalizerTask(environment, config.get('groundTruth'))
print 'Setting up Experiment'
experiment = Experiment(task, agent)
i = 0
print 'Main Loop'
while i < config.geti('maximumEpochs'):
  print 'Epoch',i,'(epsilon:{:5.3f})'.format(epsilon)
  experiment.doInteractions(int(config.get('numInteractions')))
  agent.learn()
  agent.reset()
  i += 1
  epsilon = adjustEpsilon(config.geti('maximumEpochs'), i, epsilon)
  controller.setEpsilonGreedy(epsilon)
