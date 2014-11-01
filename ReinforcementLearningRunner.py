import os, sys
import RLConfig as config
import utils as cu

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

class ReinforcementLearningRunner():

  def __init__(self, mode):
    self.mode = mode
    cu.mem('Reinforcement Learning Started')
    self.environment = RegionFilteringEnvironment(config.get(mode+'Database'), mode)
    self.controller = QNetwork()
    cu.mem('QNetwork controller created')
    self.learner = None
    self.agent = RegionFilteringAgent(self.controller, self.learner)
    self.task = RegionFilteringTask(self.environment, config.get(mode+'GroundTruth'))
    self.experiment = Experiment(self.task, self.agent)

  def runEpoch(self, interactions):
    img = 0
    while img < len(self.environment.db.images):
      self.experiment.doInteractions(interactions)
      self.agent.learn()
      self.agent.reset()
      self.environment.loadNextEpisode()
      img += 1

  def run(self):
    if self.mode == 'train':
      self.agent.persistMemory = True
      self.agent.startReplayMemory(len(self.environment.db.images), config.geti('trainInteractions'), config.geti('stateFeatures'))
      self.train()
    elif self.mode == 'test':
      self.agent.persistMemory = False
      self.test()

  def train(self):
    interactions = config.geti('trainInteractions')
    epsilon = 1.0
    self.controller.setEpsilonGreedy(epsilon)
    print 'Epoch 0: Exploration'
    self.runEpoch(interactions)
    self.learner = QLearning()
    self.agent.learner = self.learner
    epoch = 1
    egEpochs = config.geti('epsilonGreedyEpochs')
    while epoch <= egEpochs:
      epsilon = epsilon - 1/egEpochs if epsilon > 0.1 else 0.1
      self.controller.setEpsilonGreedy(epsilon)
      print 'Epoch',epoch ,'(epsilon-greedy:{:5.3f})'.format(epsilon)
      self.runEpoch(interactions)
      epoch += 1
    epoch = 1
    maxEpochs = config.geti('exploitLearningEpochs')
    while epoch <= maxEpochs:
      print 'Epoch',epoch+egEpochs,'(exploitation mode: epsilon=0.1)'
      self.runEpoch(interactions)
      epoch += 1

  def test(self):
    interactions = config.geti('testInteractions')
    self.controller.setEpsilonGreedy(0.0)
    self.runEpoch(interactions)
  
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print 'Use: ReinforcementLearningRunner.py configFile'
    sys.exit()

  ## Load Global Configuration
  config.readConfiguration(sys.argv[1])

  from RegionFilteringEnvironment import RegionFilteringEnvironment
  from QNetwork import QNetwork
  from QLearning import QLearning
  from RegionFilteringTask import RegionFilteringTask
  from RegionFilteringAgent import RegionFilteringAgent

  ## Run Training and Testing
  rl = ReinforcementLearningRunner('train')
  rl.run()
  rl = ReinforcementLearningRunner('test')
  rl.run()

