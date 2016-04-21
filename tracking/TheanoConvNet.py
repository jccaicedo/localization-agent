import theano as Theano
import theano.tensor as Tensor
import theano.tensor.nnet as NN
import logging
import json
#import TheanoGruRnn as tgr
import TheanoTransformerGru as tgr

from collections import OrderedDict

def initializeConv2d(use_cudnn=False):
    conv2d = NN.conv2d
    if use_cudnn and Theano.config.device[:3] == 'gpu':
        import theano.sandbox.cuda.dnn as CUDNN
        if CUDNN.dnn_available():
            logging.warning('Using CUDNN instead of Theano conv2d')
            conv2d = CUDNN.dnn_conv
    return conv2d

class TheanoConvNet(object):
    
    def __init__(self, modelArch, convFilters, useCUDNN=False):
        arch = json.load(open('architectures.json','r'))
        filters = str(convFilters)
        self.input = arch[modelArch][filters]['input']
        self.cnn = arch[modelArch][filters]['layers']
        self.output = arch[modelArch][filters]['output']
        self.conv2d = initializeConv2d(useCUDNN)


    def buildCNN(self, img, params):
        '''Builds a stacked sequential CNN using the config dict. Drawback: How to skip?'''
        act = img
        #Valid is the default
        pad = self.cnn.get('pad', 'valid')
        #TODO: validate key and values of same len
        layerKeys = sorted([key for key in self.cnn.keys() if key.startswith('conv')])
        if not (len(layerKeys) == len(params)):
            raise Exception('Layers length differs from parameters length: {} != {}'.format(len(layerKeys), len(params)))
        for layerIndex, layerKey in enumerate(layerKeys):
            layerConfig = self.cnn[layerKey]
            fmap = self.conv2d(act, params[layerIndex], subsample=(layerConfig['stride'], layerConfig['stride']), border_mode=pad)
            act = Tensor.nnet.relu(fmap)
        return act

    def initCNN(self, channels, initialValues={}):
        layerKeys = sorted([key for key in self.cnn.keys() if key.startswith('conv')])
        params = []
        for layerIndex, layerKey in enumerate(layerKeys):
            layerConfig = self.cnn[layerKey]
            inputChannels = self.cnn[layerKeys[layerIndex-1]]['filters'] if layerIndex > 0 else channels
            convParam = Theano.shared(initialValues[layerKey] if layerKey in initialValues else tgr.glorot_uniform((layerConfig['filters'], inputChannels, layerConfig['size'], layerConfig['size'])), name=layerKey)
            params += [convParam]
        return params

