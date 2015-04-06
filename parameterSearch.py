import shutil
import subprocess
import sys
import os

import RLConfig as config

if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print sys.argv[0], 'configTemplatePath', 'configDir', 'outputDir'
        sys.exit()
    configTemplatePath = sys.argv[1]
    configDir = sys.argv[2]
    outputDir = sys.argv[3]
    learningRates = [0.01, 0.001, 0.0001, 0.00001]
    explorationEpochsList = [5]
    epsilonGreedyEpochsList = [0]
    exploitLearningEpochsList = [0]
    categories = ['Tiger1', 'Doll', 'Basketball']
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
                                aConfig = configTemplate.format(**parametersDict)
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
        if os.exists(testMemory):
            print 'Removing {}'.format(testMemory)
            shutil.rmtree(testMemory)
        for fileName in os.listdir(networkDir):
            if fileName.startswith(snapshotPrefix):
                print 'Removing {}'.format(fileName)
                os.remove(os.path.join(networkDir, fileName))
        outFile = open(os.path.join(outputDir, configName + '.out'), 'w')
        errFile = open(os.path.join(outputDir, configName + '.err'), 'w')
        process = subprocess.Popen(['python', 'BoxSearchRunner.py', configPath], stdout=outFile, stderr=errFile)
        process.wait()
