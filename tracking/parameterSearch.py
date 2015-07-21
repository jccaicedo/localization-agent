import shutil
import subprocess
import sys
import os

import learn.rl.RLConfig as config

if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print sys.argv[0], 'configTemplatePath', 'configDir', 'outputDir'
        sys.exit()
    configTemplatePath = sys.argv[1]
    configDir = sys.argv[2]
    outputDir = sys.argv[3]
    learningRates = [0.001]
    explorationEpochsList = [1]
    epsilonGreedyEpochsList = [1]
    exploitLearningEpochsList = [0]
    categories = ['FramePairTracker']
    basePath = '/home/jccaicedo/data/tracking/exp01/'
    trainingIterationsPerBatches = [10]
    trainingBatchSizes = [32]

    #generate configs
    configTemplate = open(configTemplatePath, 'r').read()
    for learningRate in learningRates:
        for explorationEpochs in explorationEpochsList:
            for epsilonGreedyEpochs in epsilonGreedyEpochsList:
                for exploitLearningEpochs in exploitLearningEpochsList:
                    for category in categories:
                        for trainingIterationsPerBatch in trainingIterationsPerBatches:
                            for trainingBatchSize in trainingBatchSizes:
                                parametersDict = {'learningRate': learningRate, 'explorationEpochs': explorationEpochs, 'epsilonGreedyEpochs': epsilonGreedyEpochs, 'exploitLearningEpochs': exploitLearningEpochs, 'category': category, 'trainingIterationsPerBatch': trainingIterationsPerBatch, 'trainingBatchSize': trainingBatchSize}
                                aConfig = configTemplate.format(basePath=basePath, **parametersDict)
                                configName = 'rl{}.config'.format('_'.join(['{key}{value}'.format(key=key, value=value) for key, value in parametersDict.iteritems()]))
                                configPath = os.path.join(configDir, configName)
                                print 'Generating config file {}'.format(configPath)
                                outputConfig = open(configPath, 'w')
                                outputConfig.write(aConfig)
                                outputConfig.close()

    #run experiments
    configNames = os.listdir(configDir)
    for configName in configNames:
        configPath = os.path.join(configDir, configName)
        print 'Reading {} config'.format(configPath)
        config.readConfiguration(configPath)
        #erase models and memory
        networkDir = config.get('networkDir')
        snapshotPrefix = config.get('snapshotPrefix')
        testMemory = config.get('testMemory')
        if os.path.exists(testMemory):
            print 'Removing {}'.format(testMemory)
            shutil.rmtree(testMemory)
        os.mkdir(testMemory)
        for fileName in os.listdir(networkDir):
            if fileName.startswith(snapshotPrefix):
                print 'Removing {}'.format(fileName)
                os.remove(os.path.join(networkDir, fileName))
        outFile = open(os.path.join(outputDir, configName + '.out'), 'w')
        errFile = open(os.path.join(outputDir, configName + '.err'), 'w')
        process = subprocess.Popen(['time', 'python', 'run.py', 'Tracker', configPath, 'train'], stdout=outFile, stderr=errFile)
        process.wait()
        if not process.returncode == 0:
            print 'Aborting! Return code for {}: {}'.format(configName, process.returncode)
            sys.exit()
