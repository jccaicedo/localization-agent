import os, sys
import RLConfig as config
import utils as cu

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

class BoxSearchRunner():

  def __init__(self, mode):
    self.mode = mode
    cu.mem('Reinforcement Learning Started')
    self.environment = BoxSearchEnvironment(config.get(mode+'Database'), mode)
    self.controller = QNetwork()
    cu.mem('QNetwork controller created')
    self.learner = None
    self.agent = BoxSearchAgent(self.controller, self.learner)
    self.task = BoxSearchTask(self.environment, config.get(mode+'GroundTruth'))
    self.experiment = Experiment(self.task, self.agent)

  def runEpoch(self, interactions, maxImgs):
    img = 0
    s = cu.tic()
    while img < maxImgs:
      self.experiment.doInteractions(interactions)
      self.agent.learn()
      self.agent.reset()
      self.environment.loadNextEpisode()
      img += 1
    s = cu.toc('Run epoch with ' + str(maxImgs) + ' episodes', s)

  def run(self):
    if self.mode == 'train':
      self.agent.persistMemory = True
      self.agent.startReplayMemory(len(self.environment.imageList), config.geti('trainInteractions'), config.geti('stateFeatures'))
      self.train()
    elif self.mode == 'test':
      self.agent.persistMemory = False
      self.test()

  def train(self):
    interactions = config.geti('trainInteractions')
    minEpsilon = config.getf('minTrainingEpsilon')
    epochSize = len(self.environment.imageList)/10
    epsilon = 1.0
    self.controller.setEpsilonGreedy(epsilon)
    print 'Epoch 0: Exploration'
    self.runEpoch(interactions, len(self.environment.imageList))
    self.learner = QLearning()
    self.agent.learner = self.learner
    epoch = 1
    egEpochs = config.geti('epsilonGreedyEpochs')
    while epoch <= egEpochs:
      epsilon = epsilon - (1.0-minEpsilon)/float(egEpochs) 
      if epsilon < minEpsilon: epsilon = minEpsilon
      self.controller.setEpsilonGreedy(epsilon)
      print 'Epoch',epoch ,'(epsilon-greedy:{:5.3f})'.format(epsilon)
      self.runEpoch(interactions, epochSize)
      epoch += 1
    epoch = 1
    maxEpochs = config.geti('exploitLearningEpochs')
    while epoch <= maxEpochs:
      print 'Epoch',epoch+egEpochs,'(exploitation mode: epsilon={:5.3f})'.format(epsilon)
      self.runEpoch(interactions, epochSize)
      epoch += 1

  def test(self):
    interactions = config.geti('testInteractions')
    self.controller.setEpsilonGreedy(config.getf('testEpsilon'))
    self.runEpoch(interactions, len(self.environment.imageList))
  
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print 'Use: ReinforcementLearningRunner.py configFile'
    sys.exit()

  ## Load Global Configuration
  config.readConfiguration(sys.argv[1])

  from QNetwork import QNetwork
  from QLearning import QLearning
  from BoxSearchEnvironment import BoxSearchEnvironment
  from BoxSearchTask import BoxSearchTask
  from BoxSearchAgent import BoxSearchAgent

  ## Run Training and Testing
  rl = BoxSearchRunner('train')
  rl.run()
  rl = BoxSearchRunner('test')
  rl.run()

