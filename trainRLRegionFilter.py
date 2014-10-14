import os, sys
import RLConfig as config
if len(sys.argv) < 2:
  print 'Use: trainRLRegionFilter.py configFile'
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

from RegionFilteringEnvironment import RegionFilteringEnvironment
from QNetwork import QNetwork
from QLearning import QLearning
from RegionFilteringTask import RegionFilteringTask
from RegionFilteringAgent import RegionFilteringAgent

print 'Starting Environment'
epsilon = 1.0
environment = RegionFilteringEnvironment(config.get('listOfImages'), config.get('featuresDir'), 'Training')
print 'Initializing QNetwork'
controller = QNetwork()
#controller.setEpsilonGreedy(epsilon)
print 'Initializing Q Learner'
learner = QLearning()
print 'Preparing Agent'
agent = RegionFilteringAgent(controller, learner)
print 'Configuring Task'
task = RegionFilteringTask(environment, config.get('groundTruth'))
print 'Setting up Experiment'
experiment = Experiment(task, agent)
i = 0
print 'Main Loop'
while i < config.geti('maximumEpochs'):
  print 'Epoch',i #,'(epsilon:{:5.3f})'.format(epsilon)
  experiment.doInteractions(config.geti('numInteractions'))
  agent.learn()
  agent.reset()
  i += 1
  #epsilon = adjustEpsilon(config.geti('maximumEpochs'), i, epsilon)
  #controller.setEpsilonGreedy(epsilon)
