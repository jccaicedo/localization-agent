import os,sys
import subprocess
import time
import Image

import RLConfig as config

class CaffeMultiLayerPerceptronManagement():
  
  def __init__(self, workingDir):
    self.directory = workingDir
    self.step = config.geti('trainingIterationsPerBatch')
    self.readCheckpoint()
    self.writeSolverFile(self.step)

  def readCheckpoint(self):
    checkpointFile = self.directory + '/CHECKPOINT.VAR'
    if os.path.isfile(checkpointFile):
      self.checkpoint = int(open(checkpointFile).readline())
    else:
      self.checkpoint = 0

  def writeCheckpoint(self):
    checkpointFile = self.directory + '/CHECKPOINT.VAR'
    out = open(checkpointFile,'w')
    out.write( str(self.checkpoint) )
    out.close()

  def doNetworkTraining(self):
    if self.checkpoint == 0:
      # launch finetuning
      self.runNetworkTuning(config.get('pretrainedModel'))
    else:
      # Resume training
      self.runNetworkTuning(config.get('snapshotPrefix')+'_iter_'+str(self.step))

    self.checkpoint += self.step
    self.writeCheckpoint()
    
  def writeSolverFile(self, maxIter):
    out = open(self.directory + '/solver.prototxt','w')
    out.write('train_net: "train.prototxt"\n')
    out.write('test_net: "val.prototxt"\n')
    out.write('test_iter: 1000\n')
    out.write('test_interval: 5000\n')
    out.write('base_lr: 0.001\n')
    out.write('lr_policy: "step"\n')
    out.write('gamma: 0.15\n')
    out.write('stepsize: 20000\n')
    out.write('display: 10\n')
    out.write('max_iter: ' + str(maxIter) + '\n')
    out.write('momentum: 0.9\n')
    out.write('weight_decay: 0.0005\n')
    out.write('snapshot: ' + str(maxIter) + '\n')
    out.write('snapshot_prefix: "multilayer_qlearner"\n')
    out.close()

  def readTrainingDatabase(self, file):
    records = []
    if os.path.isfile(self.directory + file):
      data = [x.split() for x in open(self.directory + file)]
      for d in data:
        records.append( map(float, d) )
    return records, len(records)

  def saveDatabaseFile(self, records, outputFile):
    out = open(self.directory + outputFile, 'w')
    for r in records:
      out.write( str(int(r[0])) + ' ' + ' '.join(map(str,r[1:])) + '\n' )
    out.close()
    return len(records)

  def runNetworkTraining(self, args):
    my_env = os.environ.copy()
    my_env['GLOG_logtostderr']='1'
    my_env['GLOG_minloglevel']='0'
    monitorFile = self.directory + '/' + config.get('snapshotPrefix')+'_iter_'+str(self.checkpoint+self.step)+'.solverstate'
    p = subprocess.Popen(args, env=my_env, cwd=self.directory)
    while not os.path.isfile(monitorFile):
      time.sleep(10)
    p.terminate()
    if self.checkpoint > 0:
      os.remove(self.directory + '/' + config.get('snapshotPrefix')+'_iter_'+str(self.checkpoint)+'.solverstate')
      os.remove(self.directory + '/' + config.get('snapshotPrefix')+'_iter_'+str(self.checkpoint))
    return

  def runNetworkTuning(self, pretrained):
    my_env = os.environ.copy()
    my_env['GLOG_logtostderr']='1'
    my_env['GLOG_minloglevel']='0'
    args = [config.get('tools') + '/finetune_net.bin', config.get('solverFile'), pretrained]
    p = subprocess.Popen(args, env=my_env, cwd=self.directory)
    p.wait()
    return
