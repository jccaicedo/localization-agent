import os, sys
import learn.rl.RLConfig as config
import utils.utils as cu

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

import shutil

class TrackerRunner():

  def __init__(self, mode):
    self.mode = mode
    cu.mem('Reinforcement Learning Started')
    self.environment = TrackerAugmentedEnvironment(config.get(mode+'Database'), mode)
    self.controller = QNetwork()
    cu.mem('QNetwork controller created')
    self.learner = None
    self.agent = TrackerAgent(self.controller, self.learner)
    self.task = TrackerTask(self.environment)
    self.experiment = Experiment(self.task, self.agent)

  def runEpoch(self, interactions):
    episode = 0
    s = cu.tic()
    while episode < self.environment.numEpisodes():
      while not self.environment.episodeDone:
        k = 0
        self.environment.loadNextFrame()
        while k < interactions and not self.environment.landmarkFound:
          self.experiment._oneInteraction()
          k += 1
        self.task.displayEpisodePerformance()
        self.agent.learn()
      self.agent.reset()
      self.environment.loadNextEpisode()
      episode += 1
    self.environment.epochDone()
    s = cu.toc('Run epoch with ' + str(episode) + ' episodes', s)

  def run(self):
    if self.mode == 'train':
      self.agent.persistMemory = True
      self.agent.startReplayMemory(len(self.environment.episodes[0].imageList)*self.environment.numEpisodes(), config.geti('trainInteractions'))
      self.train()
    elif self.mode == 'test':
      self.agent.persistMemory = False
      self.test()

  def train(self):
    networkFile = config.get('networkDir') + config.get('snapshotPrefix') + '_iter_' + config.get('trainingIterationsPerBatch') + '.caffemodel'
    interactions = config.geti('trainInteractions')
    minEpsilon = config.getf('minTrainingEpsilon')
    epsilon = 1.0
    self.controller.setEpsilonGreedy(epsilon, self.environment.sampleAction)
    epoch = 1
    exEpochs = config.geti('explorationEpochs')
    while epoch <= exEpochs:
      s = cu.tic()
      print 'Epoch',epoch,': Exploration (epsilon=1.0)'
      self.runEpoch(interactions)
      self.task.flushStats()
      s = cu.toc('Epoch done in ',s)
      epoch += 1
    self.learner = QLearning()
    self.agent.learner = self.learner
    egEpochs = config.geti('epsilonGreedyEpochs')
    while epoch <= egEpochs + exEpochs:
      s = cu.tic()
      epsilon = epsilon - (1.0-minEpsilon)/float(egEpochs)
      if epsilon < minEpsilon: epsilon = minEpsilon
      self.controller.setEpsilonGreedy(epsilon, self.environment.sampleAction)
      print 'Epoch',epoch ,'(epsilon-greedy:{:5.3f})'.format(epsilon)
      self.runEpoch(interactions)
      self.task.flushStats()
      self.doValidation(epoch)
      s = cu.toc('Epoch done in ',s)
      epoch += 1
    maxEpochs = config.geti('exploitLearningEpochs') + exEpochs + egEpochs
    while epoch <= maxEpochs:
      s = cu.tic()
      print 'Epoch',epoch,'(exploitation mode: epsilon={:5.3f})'.format(epsilon)
      self.runEpoch(interactions)
      self.task.flushStats()
      self.doValidation(epoch)
      s = cu.toc('Epoch done in ',s)
      shutil.copy(networkFile, networkFile + '.' + str(epoch))
      epoch += 1

  def test(self):
    interactions = config.geti('testInteractions')
    self.controller.setEpsilonGreedy(config.getf('testEpsilon'))
    self.runEpoch(interactions)

  def doValidation(self, epoch):
    if epoch % config.geti('validationEpochs') != 0:
      return
    auxRL = TrackerRunner('test')
    auxRL.run()
    indexType = config.get('evaluationIndexType')
    category = config.get('category')
    if indexType == 'pascal':
      categories, catIndex = te.get20Categories()
    elif indexType == 'relations':
      categories, catIndex = te.getCategories()
    elif indexType == 'finetunedRelations':
      categories, catIndex = te.getRelationCategories()
    if category in categories:
        catI = categories.index(category)
    else:
        catI = -1
    scoredDetections = te.loadScores(config.get('testMemory'), catI)
    groundTruth = auxRL.environment.getGroundTruth()
    pl,rl = te.evaluateCategory(scoredDetections, 'landmarks', groundTruth)
    line = lambda x,y,z: x + '\t{:5.3f}\t{:5.3f}\n'.format(y,z)
    #print line('Validation Scores:',ps,rs)
    print line('Validation Landmarks:',pl,rl)
  
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print 'Use: {} configFile'.format(sys.argv[0])
    sys.exit()

  ## Load Global Configuration
  config.readConfiguration(sys.argv[1])

  from QNetwork import QNetwork
  from QLearning import QLearning
  from TrackerAugmentedEnvironment import TrackerAugmentedEnvironment
  from TrackerTask import TrackerTask
  from TrackerAgent import TrackerAgent
  import TrackerEvaluation as te

  if len(sys.argv) == 2:
    ## Run Training and Testing
    rl = TrackerRunner('train')
    rl.run()
    rl = TrackerRunner('test')
    rl.run()
  elif len(sys.argv) == 3:
    rl = TrackerRunner(sys.argv[2])
    rl.run()
