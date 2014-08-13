__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner

class DeepQLearning(ValueBasedLearner):

  offPolicy = True
  batchMode = True
  dataset = []

  def __init__(self, alpha=0.5, gamma=0.99):
    ValueBasedLearner.__init__(self)
    self.alpha = alpha
    self.gamma = gamma

  def learn(self, data):
    images = []
    for d in data:
      images.append(d[0])
      self.dataset.append(d)
      print d
    print len(data),len(set(images))
    pass
