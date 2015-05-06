import os, sys
import learn.rl.RLConfig as config
import utils.utils as cu

from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA
from pybrain.rl.experiments import Experiment

import shutil

class BoxSearchRunner():

  def __init__(self, mode):
    self.mode = mode
    cu.mem('Reinforcement Learning Started')
    self.environment = BoxSearchEnvironment(config.get(mode+'Database'), mode, config.get(mode+'GroundTruth'))
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
      k = 0
      while not self.environment.episodeDone and k < interactions:
        self.experiment._oneInteraction()
        k += 1
      self.agent.learn()
      self.agent.reset()
      self.environment.loadNextEpisode()
      img += 1
    s = cu.toc('Run epoch with ' + str(maxImgs) + ' episodes', s)

  def run(self):
    if self.mode == 'train':
      self.agent.persistMemory = True
      self.agent.startReplayMemory(len(self.environment.imageList), config.geti('trainInteractions'))
      #self.agent.assignPriorMemory(self.environment.priorMemory)
      self.train()
    elif self.mode == 'test':
      self.agent.persistMemory = False
      self.test()

  def train(self):
    networkFile = config.get('networkDir') + config.get('snapshotPrefix') + '_iter_' + config.get('trainingIterationsPerBatch') + '.caffemodel'
    interactions = config.geti('trainInteractions')
    minEpsilon = config.getf('minTrainingEpsilon')
    epochSize = len(self.environment.imageList)/1
    epsilon = 1.0
    self.controller.setEpsilonGreedy(epsilon, self.environment.sampleAction)
    epoch = 1
    exEpochs = config.geti('explorationEpochs')
    while epoch <= exEpochs:
      s = cu.tic()
      print 'Epoch',epoch,': Exploration (epsilon=1.0)'
      self.runEpoch(interactions, len(self.environment.imageList))
      self.task.flushStats()
      self.doValidation(epoch)
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
      self.runEpoch(interactions, epochSize)
      self.task.flushStats()
      self.doValidation(epoch)
      s = cu.toc('Epoch done in ',s)
      epoch += 1
    maxEpochs = config.geti('exploitLearningEpochs') + exEpochs + egEpochs
    while epoch <= maxEpochs:
      s = cu.tic()
      print 'Epoch',epoch,'(exploitation mode: epsilon={:5.3f})'.format(epsilon)
      self.runEpoch(interactions, epochSize)
      self.task.flushStats()
      self.doValidation(epoch)
      s = cu.toc('Epoch done in ',s)
      shutil.copy(networkFile, networkFile + '.' + str(epoch))
      epoch += 1

  def test(self):
    interactions = config.geti('testInteractions')
    self.controller.setEpsilonGreedy(config.getf('testEpsilon'))
    self.runEpoch(interactions, len(self.environment.imageList))

  def doValidation(self, epoch):
    if epoch % config.geti('validationEpochs') != 0:
      return
    auxRL = BoxSearchRunner('test')
    auxRL.run()
    indexType = config.get('evaluationIndexType')
    category = config.get('category')
    if indexType == 'pascal':
      categories, catIndex = bse.get20Categories()
    elif indexType == 'relations':
      categories, catIndex = bse.getCategories()
    elif indexType == 'finetunedRelations':
      categories, catIndex = bse.getRelationCategories()
    if category in categories:
        catI = categories.index(category)
    else:
        catI = -1
    scoredDetections = bse.loadScores(config.get('testMemory'), catI)
    groundTruthFile = config.get('testGroundTruth')
    #ps,rs = bse.evaluateCategory(scoredDetections, 'scores', groundTruthFile)
    pl,rl = bse.evaluateCategory(scoredDetections, 'landmarks', groundTruthFile)
    line = lambda x,y,z: x + '\t{:5.3f}\t{:5.3f}\n'.format(y,z)
    #print line('Validation Scores:',ps,rs)
    print line('Validation Landmarks:',pl,rl)


#def main():
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
  import BoxSearchEvaluation as bse

  print 'Hello'

  if len(sys.argv) == 2:
    ## Run Training and Testing
    rl = BoxSearchRunner('train')
    rl.run()
    rl = BoxSearchRunner('test')
    rl.run()
  elif len(sys.argv) == 3:
    # Run only the requested mode
    rl = BoxSearchRunner(sys.argv[2])
    rl.run()


