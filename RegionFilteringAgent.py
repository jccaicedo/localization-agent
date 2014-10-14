__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

import RLConfig as config

import numpy as np

class RegionFilteringAgent():

  image = None
  observation = None
  action = None
  reward = None
  cumReward = 0
  memory = []
  t = 0
  
  def __init__(self, qnet, learner=None):
    self.controller = qnet
    self.learner = learner
    self.obsQueue = []

  def integrateObservation(self, obs):
    self.observation = obs[1]
    self.action = None
    self.reward = None
    if 'NewImage' in self.observation.keys():
      self.obsQueue = []
    print 'Agent::integrateObservation'

  def getAction(self):
    assert self.observation != None
    assert self.action == None
    assert self.reward == None

    print 'Agent::getAction'

    if "NoChange" not in self.observation.keys(): # Last movement did not produce a new observation
      self.t += 1
      relationScores = self.observation['features'].shape[1]
      ims = self.observation['imgSize']
      boxCoords = 4
      area = 1
      outputs = 10 # config.something
      boxes = self.observation['features'].shape[0]
      Z = np.zeros( (boxes, 2*relationScores + 2*(boxCoords + area) + outputs) ) 
      ## INPUTS FOR THE NETWORK
      ## 1. Normalized box coordinates
      Z[:,0:boxCoords] = [ [b[0]/ims[0], b[1]/ims[1], b[2]/ims[0], b[3]/ims[1]] for b in self.observation['boxes'] ]
      ## 2. Area of box relative to the whole image
      Z[:,boxCoords] = self.observation['areas']
      ## 3. Relation scores for the box
      Z[:,boxCoords+area:boxCoords+area+relationScores] = self.observation['features']
      ## 4. Box coordinates of previous box
      rec = boxCoords + area + relationScores
      rb = self.observation['rootBox']
      rb = [rb[0]/ims[0], rb[1]/ims[1], rb[2]/ims[0], rb[3]/ims[1]]
      Z[:,rec:rec+boxCoords] = np.tile( rb, (boxes,1))
      # 5. Area of previous box
      Z[:,rec+boxCoords:rec+boxCoords+area] = np.tile( self.observation['rootArea'], (boxes,1))
      # 6. Previous box relation scores
      Z[:,rec+boxCoords+area:rec+boxCoords+area+relationScores] = np.tile( self.observation['rootFeatures'][0,:], (boxes,1))
      # 7. Encoding of previous action
      rec += boxCoords + area + relationScores
      action1ofk = [0 for a in range(outputs)]
      action1ofk[self.observation['lastAction']] = 1.0
      Z[:,rec:rec+outputs] = np.tile(action1ofk, (boxes,1))
      # 8. Time to live
      #Z[:,-1] = float(self.t) #/20 #config.MAX_STEPS_ALLOWED
      print Z.shape

      values = self.controller.getActionValues( Z )
      actions = np.argmax(values, 1)
      maxValues = np.max(values, 1)
      for i in range(actions.shape[0]):
        self.obsQueue.append( [self.observation['indexes'][i], actions[i], maxValues[i]] + Z[i,:].tolist() )
      self.obsQueue.sort(key=lambda x: x[2], reverse=True)

    print 'AgentQueue=>',[x[0] for x in self.obsQueue]
    if len(self.obsQueue) > 0:
      self.action = self.obsQueue.pop(0)
    else:
      self.action = 'terminate'
    return self.action

  def giveReward(self, r):
    print 'Agent::giveReward=>',r
    assert self.observation != None
    assert self.action != None
    assert self.reward == None

    self.reward = r
    self.cumReward += sum(r)

    # Replay Memory Format: boxIndex action maxValue features[62] reward
    if self.action != 'terminate':
      self.memory.append( self.action + r )

  def reset(self):
    print 'Agent::reset'
    self.observation = None
    self.action = None
    self.reward = None
    self.obsQueue = []
    self.memory = []
    self.controller.loadNetwork()
    self.t = 0

  def learn(self):
    print 'Agent:learn:'
    if self.learner != None:
      self.learner.learn(self.memory, self.controller)

