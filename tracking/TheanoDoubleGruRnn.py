import theano as Theano
import theano.tensor as Tensor
import numpy as NP
import numpy.random as RNG
import theano.tensor.nnet as NN
import cPickle as pickle
import VisualAttention
import logging

from collections import OrderedDict
from LasagneVGG16 import LasagneVGG16


def smooth_l1(x):
    return Tensor.switch(Tensor.lt(Tensor.abs_(x),1), 0.5*x**2, Tensor.abs_(x)-0.5)

def l2(x):
    return x ** 2

def box2cwh(boxTensor):
    xc = (boxTensor[:,:,2]+boxTensor[:,:,0])/2
    yc = (boxTensor[:,:,3]+boxTensor[:,:,1])/2
    width = (boxTensor[:,:,2]-boxTensor[:,:,0])
    height = (boxTensor[:,:,3]-boxTensor[:,:,1])
    return Tensor.stacklists([xc,yc,width,height]).dimshuffle(1,2,0)

#TODO: turn into GRU class
def gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg):
    gru_in = Tensor.reshape(features, (features.shape[0], Tensor.prod(features.shape[1:])))
    #gru_in = Tensor.concatenate([flat1, prev_bbox], axis=1) #TODO: Remove this thing!
    gru_z = NN.sigmoid(Tensor.dot(gru_in, Wz) + Tensor.dot(state, Uz) + bz)
    gru_r = NN.sigmoid(Tensor.dot(gru_in, Wr) + Tensor.dot(state, Ur) + br)
    gru_h_ = Tensor.tanh(Tensor.dot(gru_in, Wg) + Tensor.dot(gru_r * state, Ug) + bg)
    gru_h = (1-gru_z) * state + gru_z * gru_h_
    return gru_h

def boxRegressor(gru_h, W_fc, b_fc):
    bbox = Tensor.tanh(Tensor.dot(gru_h, W_fc) + b_fc)
    return bbox
    
def initGru(inputDim, stateDim, level):
    Wr = Theano.shared(glorot_uniform((inputDim, stateDim)), name='Wr'+level)
    Ur = Theano.shared(orthogonal((stateDim, stateDim)), name='Ur'+level)
    br = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='br'+level)
    Wz = Theano.shared(glorot_uniform((inputDim, stateDim)), name='Wz'+level)
    Uz = Theano.shared(orthogonal((stateDim, stateDim)), name='Uz'+level)
    bz = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='bz'+level)
    Wg = Theano.shared(glorot_uniform((inputDim, stateDim)), name='Wg'+level)
    Ug = Theano.shared(orthogonal((stateDim, stateDim)), name='Ug'+level)
    bg = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='bg'+level)
    return Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg

def initRegressor(stateDim, targetDim, zeroTailFc):
    if not zeroTailFc:
        W_fcinit = glorot_uniform((stateDim, targetDim))
    else:
        W_fcinit = NP.zeros((stateDim, targetDim), dtype=Theano.config.floatX)
    W_fc = Theano.shared(W_fcinit, name='W_fc')
    b_fc = Theano.shared(NP.zeros((targetDim,), dtype=Theano.config.floatX), name='b_fc')
    return W_fc, b_fc

def buildCNN(img, cnnConfig, params):
    '''Builds a stacked sequential CNN using the config dict. Drawback: How to skip?'''
    act = img
    #Valid is the default
    pad = cnnConfig.get('pad', 'valid')
    #TODO: validate key and values of same len
    layerKeys = sorted([key for key in cnnConfig.keys() if key.startswith('conv')])
    if not (len(layerKeys) == len(params)):
        raise Exception('Layers length differs from parameters length: {} != {}'.format(len(layerKeys), len(params)))
    for layerIndex, layerKey in enumerate(layerKeys):
        layerConfig = cnnConfig[layerKey]
        fmap = NN.conv2d(act, params[layerIndex], subsample=(layerConfig['stride'], layerConfig['stride']), border_mode=pad)
        act = Tensor.nnet.relu(fmap)
    return act

def initCNN(cnnConfig, channels, initialValues={}):
    layerKeys = sorted([key for key in cnnConfig.keys() if key.startswith('conv')])
    params = []
    for layerIndex, layerKey in enumerate(layerKeys):
        layerConfig = cnnConfig[layerKey]
        inputChannels = cnnConfig[layerKeys[layerIndex-1]]['filters'] if layerIndex > 0 else channels
        convParam = Theano.shared(initialValues[layerKey] if layerKey in initialValues else glorot_uniform((layerConfig['filters'], inputChannels, layerConfig['size'], layerConfig['size'])), name=layerKey)
        params += [convParam]
    return params

def initializeConv2d(use_cudnn=False):
    conv2d = NN.conv2d
    if use_cudnn and Theano.config.device[:3] == 'gpu':
        import theano.sandbox.cuda.dnn as CUDNN
        if CUDNN.dnn_available():
            logging.warning('Using CUDNN instead of Theano conv2d')
            conv2d = CUDNN.dnn_conv
    return conv2d

def rmsprop(cost, params, lr=0.0005, rho=0.9, epsilon=1e-6):
    '''
    Borrowed from keras, no constraints, though
    '''
    updates = OrderedDict()
    grads = Theano.grad(cost, params)
    acc = [Theano.shared(NP.zeros(p.get_value().shape, dtype=Theano.config.floatX)) for p in params]
    for p, g, a in zip(params, grads, acc):
        new_a = rho * a + (1 - rho) * g ** 2
        updates[a] = new_a
        new_p = p - lr * g / Tensor.sqrt(new_a + epsilon)
        updates[p] = new_p

    return updates  
    
    
def glorot_uniform(shape):
    '''
    Borrowed from keras
    '''
    fan_in, fan_out = get_fans(shape)
    s = NP.sqrt(6. / (fan_in + fan_out))
    return NP.cast[Theano.config.floatX](RNG.uniform(low=-s, high=s, size=shape))
    
    
def get_fans(shape):
    '''
    Borrowed from keras
    '''
    fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def orthogonal(shape, scale=1.1):
    '''
    Borrowed from keras
    '''
    flat_shape = (shape[0], NP.prod(shape[1:]))
    a = RNG.normal(0, 1, flat_shape)
    u, _, v = NP.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    
    return NP.cast[Theano.config.floatX](q)

def loadModel(modelPath):
    logging.info('Loading model from %s', modelPath)
    with open(modelPath, 'rb') as modelFile: 
        model = pickle.load(modelFile)
    if not isinstance(model, TheanoGruRnn):
        raise Exception('Model of type {}, expected {}'.format(type(model), TheanoGruRnn))
    return model
      
def saveModel(model, modelPath):
    #TODO: silent for trax
    if not isinstance(model, TheanoGruRnn):
        raise Exception('Model of type {}, expected {}'.format(type(model), TheanoGruRnn))
    print 'Saving model to {}'.format(modelPath)
    with open(modelPath, 'wb') as modelFile:
        pickle.dump(model, modelFile)
        
def getTensor(name, dtype, dim):
    if dtype == None:
        dtype = Theano.config.floatX
    
    return Tensor.TensorType(dtype, [False] * dim, name=name)()

class TheanoGruRnn(object):
    
    fitFunc = None
    forwardFunc = None
    params = None
    seqLength = None
    stepFunc = None
    
    def __init__(self, inputDim, stateDim, targetDim, batchSize, seqLength, zeroTailFc, learningRate, use_cudnn, imgSize, modelArch='oneConvLayers', norm=l2, useAttention=False, modelPath=None, layerKey=None, convFilters=32, computeFlow=False):
        ### Computed hyperparameters begin
        self.modelArch = modelArch
        if self.modelArch == 'oneConvLayers':
            self.cnn = {'conv1':{'filters':32, 'size':10, 'stride':5, 'output':(((inputDim-10)/5+1)**2)*32 }}
            inputDim = self.cnn['conv1']['output']
        elif self.modelArch == 'lasagne':
            self.cnn = LasagneVGG16(modelPath, layerKey)
            inputDim = 512 * 7 * 7
        elif self.modelArch == 'twoConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((128-5)/2+1)**2)*64 },
                            'conv2':{'filters':32, 'size':3, 'stride':2, 'output':(((62-3)/2+1)**2)*32 }}
                inputDim = self.cnn['conv2']['output']
            elif convFilters == 2:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((128-5)/2+1)**2)*64 },
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(((62-3)/2+1)**2)*64 }}
                inputDim = self.cnn['conv2']['output']
            elif convFilters == 3:
                self.cnn = {'conv1':{'filters':128, 'size':5, 'stride':2, 'output':(((128-5)/2+1)**2)*128 },
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(((62-3)/2+1)**2)*64 }}
                inputDim = self.cnn['conv2']['output']
            elif convFilters == 4:
                self.cnn = {'conv1':{'filters':128, 'size':5, 'stride':2, 'output':(((128-5)/2+1)**2)*128 },
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(((62-3)/2+1)**2)*128 }}
                inputDim = self.cnn['conv2']['output']
        elif self.modelArch == 'threeConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':32, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*32 },
                            'conv2':{'filters':32, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*32 },
                            'conv3':{'filters':32, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*32 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 2:
                self.cnn = {'conv1':{'filters':32, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*32 },
                            'conv2':{'filters':32, 'size':5, 'stride':2, 'output':(((94-5)/2+1)**2)*32 },
                            'conv3':{'filters':32, 'size':5, 'stride':2, 'output':(((45-5)/2+1)**2)*32 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 3:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*64 },
                            'conv2':{'filters':32, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*32 },
                            'conv3':{'filters':32, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*32 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 4:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*64 },
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*64 },
                            'conv3':{'filters':32, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*32 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 5:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*64 },
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*64 },
                            'conv3':{'filters':64, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*64 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 6:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*64 },
                            'conv2':{'filters':64, 'size':5, 'stride':2, 'output':(((94-5)/2+1)**2)*64 },
                            'conv3':{'filters':64, 'size':5, 'stride':2, 'output':(((45-5)/2+1)**2)*64 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 7:
                self.cnn = {'conv1':{'filters':96, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*96 },
                            'conv2':{'filters':128, 'size':5, 'stride':2, 'output':(((94-5)/2+1)**2)*128 },
                            'conv3':{'filters':64, 'size':5, 'stride':2, 'output':(((45-5)/2+1)**2)*64 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 8:
                self.cnn = {'conv1':{'filters':96, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*96 },
                            'conv2':{'filters':128, 'size':5, 'stride':2, 'output':(((94-5)/2+1)**2)*128 },
                            'conv3':{'filters':128, 'size':5, 'stride':2, 'output':(((45-5)/2+1)**2)*128 }}
                inputDim = self.cnn['conv3']['output']
        elif self.modelArch == 'fourConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':96, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*96 },
                            'conv2':{'filters':128, 'size':5, 'stride':2, 'output':(((94-5)/2+1)**2)*128 },
                            'conv3':{'filters':128, 'size':3, 'stride':1, 'output':(((45-3)/1+1)**2)*128 },
                            'conv4':{'filters':128, 'size':3, 'stride':2, 'output':(((43-3)/2+1)**2)*128 }}
                inputDim = self.cnn['conv4']['output']
        elif self.modelArch == 'fiveConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':96, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*96 },
                            'conv2':{'filters':128, 'size':5, 'stride':2, 'output':(((94-5)/2+1)**2)*128 },
                            'conv3':{'filters':128, 'size':3, 'stride':1, 'output':(((45-3)/1+1)**2)*128 },
                            'conv4':{'filters':128, 'size':5, 'stride':2, 'output':(((43-5)/2+1)**2)*128 },
                            'conv5':{'filters':128, 'size':3, 'stride':1, 'output':(((20-3)/1+1)**2)*128 },
                            'pad':'valid'}
                inputDim = self.cnn['conv5']['output']
            if convFilters == 2:
                self.cnn = {'conv1':{'filters':96, 'size':5, 'stride':2, 'output':(96**2)*96 },   # Feature map size 884,736 - Params: 5x5x3x96    =   7,200
                            'conv2':{'filters':128, 'size':5, 'stride':2, 'output':(48**2)*128 }, # Feature map size 294,912 - Params: 5x5x96x128  = 307,200
                            'conv3':{'filters':128, 'size':3, 'stride':1, 'output':(48**2)*128 }, # Feature map size 294,912 - Params: 3x3x128x128 = 147,456
                            'conv4':{'filters':128, 'size':3, 'stride':2, 'output':(24**2)*128 }, # Feature map size  73,728 - Params: 3x3x128x128 = 147,456
                            'conv5':{'filters':128, 'size':3, 'stride':1, 'output':(24**2)*128 }, # Feature map size  73,728 - Params: 3x3x128x128 = 147,456
                            'pad':'half'}                                                         #        TOTALS: 1'622,016 -                     = 756,768
                inputDim = self.cnn['conv5']['output']
            if convFilters == 3:
                self.cnn = {'conv1':{'filters':96, 'size':5, 'stride':2, 'output':(96**2)*96 },   # Feature map size 884,736 - Params: 5x5x3x96    =   7,200
                            'conv2':{'filters':256, 'size':5, 'stride':2, 'output':(48**2)*256 }, # Feature map size 589,824 - Params: 5x5x96x256  = 614,400
                            'conv3':{'filters':256, 'size':3, 'stride':1, 'output':(48**2)*256 }, # Feature map size 589,824 - Params: 3x3x256x256 = 589,824
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(24**2)*256 }, # Feature map size 147,456 - Params: 3x3x256x256 = 589,824
                            'conv5':{'filters':128, 'size':3, 'stride':1, 'output':(24**2)*128 }, # Feature map size  73,728 - Params: 3x3x256x128 = 294,912
                            'pad':'half'}                                                         #        TOTALS: 2'285,568 -                   = 2'096,160
                inputDim = self.cnn['conv5']['output']
            if convFilters == 4:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(96**2)*64 },   # Feature map size 589,824 - Params: 5x5x3x64    =   4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(48**2)*128 }, # Feature map size 294,912 - Params: 3x3x64x128  =  73,728
                            'conv3':{'filters':128, 'size':3, 'stride':1, 'output':(48**2)*128 }, # Feature map size 294,912 - Params: 3x3x128x128 = 147,456
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(24**2)*256 }, # Feature map size 147,456 - Params: 3x3x128x256 = 294,912
                            'conv5':{'filters':256, 'size':3, 'stride':2, 'output':(12**2)*256 }, # Feature map size  36,864 - Params: 3x3x256x256 = 589,824
                            'pad':'half'}                                                         #        TOTALS: 1'363,968 -                   = 1'110,720
                inputDim = self.cnn['conv5']['output']
        elif self.modelArch == 'sixConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(96**2)*64 },   # Feature map size 589,824 - Params: 5x5x3x64    =   4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(48**2)*128 }, # Feature map size 294,912 - Params: 3x3x64x128  =  73,728
                            'conv3':{'filters':128, 'size':3, 'stride':1, 'output':(48**2)*128 }, # Feature map size 294,912 - Params: 3x3x128x128 = 147,456
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(24**2)*256 }, # Feature map size 147,456 - Params: 3x3x128x256 = 294,912
                            'conv5':{'filters':256, 'size':3, 'stride':1, 'output':(24**2)*256 }, # Feature map size 147,456 - Params: 3x3x256x256 = 589,824
                            'conv6':{'filters':256, 'size':3, 'stride':2, 'output':(12**2)*256 }, # Feature map size  36,864 - Params: 3x3x256x256 = 589,824
                            'pad':'half'}                                                         #        TOTALS: 1'511,424 -                   = 1'700,544
                inputDim = self.cnn['conv6']['output']
        elif self.modelArch == 'fiveXConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(112**2)*64 },  # Feature map size 802,812 - Params: 5x5x3x64    =     4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(56**2)*128 }, # Feature map size 401,408 - Params: 3x3x64x128  =    73,728
                            'conv3':{'filters':256, 'size':3, 'stride':2, 'output':(28**2)*256 }, # Feature map size 200,704 - Params: 3x3x128x256 =   294,912
                            'conv4':{'filters':512, 'size':3, 'stride':2, 'output':(14**2)*512 }, # Feature map size 100,352 - Params: 3x3x256x512 = 1'179,648
                            'conv5':{'filters':512, 'size':3, 'stride':2, 'output':(7**2)*512 },  # Feature map size  25,088 - Params: 3x3x512x512 = 2'359,296
                            'pad':'half'}                                                         #        TOTALS: 1'530,364 -                     = 3'912,384
                inputDim = self.cnn['conv5']['output']
            if convFilters == 2:
                self.cnn = {'conv1':{'filters':32, 'size':5, 'stride':2, 'output':(112**2)*32 },  # Feature map size 401,408 - Params: 5x5x3x32  =  2,400
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(56**2)*64 },   # Feature map size 200,704 - Params: 3x3x32x64 = 18,432
                            'conv3':{'filters':32, 'size':3, 'stride':1, 'output':(56**2)*32 },   # Feature map size 100,352 - Params: 3x3x64x32 = 18,432
                            'conv4':{'filters':64, 'size':3, 'stride':2, 'output':(28**2)*64 },   # Feature map size  50,176 - Params: 3x3x32x64 = 18,432
                            'conv5':{'filters':32, 'size':3, 'stride':1, 'output':(28**2)*32 },   # Feature map size  25,088 - Params: 3x3x64x32 = 18,432
                            'pad':'half'}                                                         #        TOTALS:   777,728 -                   = 76,128
                inputDim = self.cnn['conv5']['output']
            if convFilters == 3:
                self.cnn = {'conv1':{'filters':32, 'size':5, 'stride':2, 'output':(112**2)*32 },  # Feature map size 401,408 - Params: 5x5x3x32  =  2,400
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(56**2)*64 },   # Feature map size 200,704 - Params: 3x3x32x64 = 18,432
                            'conv3':{'filters':32, 'size':3, 'stride':1, 'output':(56**2)*32 },   # Feature map size 100,352 - Params: 3x3x64x32 = 18,432
                            'conv4':{'filters':16, 'size':3, 'stride':1, 'output':(56**2)*16 },   # Feature map size  50,176 - Params: 3x3x32x16 =  4,608
                            'conv5':{'filters': 8, 'size':3, 'stride':1, 'output':(56**2)*8  },   # Feature map size  25,088 - Params: 3x3x16x8  =  1,152
                            'pad':'half'}                                                         #        TOTALS:   777,728 -                   = 45,024
                inputDim = self.cnn['conv5']['output']


        elif self.modelArch == 'sixXConvLayers':
            if convFilters == 1:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(112**2)*64 },  # Feature map size 802,816 - Params: 5x5x3x64    =     4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(56**2)*128 }, # Feature map size 401,408 - Params: 3x3x64x128  =    73,728
                            'conv3':{'filters':256, 'size':3, 'stride':1, 'output':(28**2)*256 }, # Feature map size 200,704 - Params: 3x3x128x256 =   294,912
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(28**2)*256 }, # Feature map size 200,704 - Params: 3x3x256x256 =   589,824
                            'conv5':{'filters':512, 'size':3, 'stride':2, 'output':(14**2)*512 }, # Feature map size 100,352 - Params: 3x3x256x512 = 1'179,648
                            'conv6':{'filters':512, 'size':3, 'stride':2, 'output':(7**2)*512 },  # Feature map size  25,088 - Params: 3x3x512x512 = 2'359,296
                            'pad':'half'}                                                         #        TOTALS: 1'745,072 -                     = 4'502,208
                inputDim = self.cnn['conv6']['output']
            if convFilters == 20:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(112**2)*64 },  # Feature map size 802,816 - Params: 5x5x3x64    =     4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(56**2)*128 }, # Feature map size 401,408 - Params: 3x3x64x128  =    73,728
                            'conv3':{'filters':256, 'size':3, 'stride':2, 'output':(28**2)*256 }, # Feature map size 200,704 - Params: 3x3x128x256 =   294,912
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(14**2)*256 }, # Feature map size 100,352 - Params: 3x3x256x256 =   589,824
                            'conv5':{'filters':512, 'size':3, 'stride':2, 'output':(7**2)*512 },  # Feature map size  25,088 - Params: 3x3x256x512 = 1'179,648
                            'conv6':{'filters':128, 'size':3, 'stride':1, 'output':(7**2)*128 },  # Feature map size   6,272 - Params: 3x3x512x128 =   589,824
                            'pad':'half'}                                                         #        TOTALS: 1'543,640 -                     = 2'732,736
                inputDim = self.cnn['conv6']['output']
            if convFilters == 21:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(112**2)*64 },  # Feature map size 802,816 - Params: 5x5x3x64    =     4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(56**2)*128 }, # Feature map size 401,408 - Params: 3x3x64x128  =    73,728
                            'conv3':{'filters':256, 'size':3, 'stride':2, 'output':(28**2)*256 }, # Feature map size 200,704 - Params: 3x3x128x256 =   294,912
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(14**2)*256 }, # Feature map size 100,352 - Params: 3x3x256x256 =   589,824
                            'conv5':{'filters':512, 'size':3, 'stride':2, 'output':(7**2)*512 },  # Feature map size  25,088 - Params: 3x3x256x512 = 1'179,648
                            'conv6':{'filters':256, 'size':3, 'stride':1, 'output':(7**2)*256 },  # Feature map size   6,272 - Params: 3x3x512x128 =   589,824
                            'pad':'half'}                                                         #        TOTALS: 1'543,640 -                     = 2'732,736
                inputDim = self.cnn['conv6']['output']
            if convFilters == 22:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(112**2)*64 },  # Feature map size 802,816 - Params: 5x5x3x64    =     4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(56**2)*128 }, # Feature map size 401,408 - Params: 3x3x64x128  =    73,728
                            'conv3':{'filters':256, 'size':3, 'stride':2, 'output':(28**2)*256 }, # Feature map size 200,704 - Params: 3x3x128x256 =   294,912
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(14**2)*256 }, # Feature map size 100,352 - Params: 3x3x256x256 =   589,824
                            'conv5':{'filters':512, 'size':3, 'stride':2, 'output':(7**2)*512 },  # Feature map size  25,088 - Params: 3x3x256x512 = 1'179,648
                            'conv6':{'filters':512, 'size':3, 'stride':1, 'output':(7**2)*512 },  # Feature map size   6,272 - Params: 3x3x512x128 =   589,824
                            'pad':'half'}                                                         #        TOTALS: 1'543,640 -                     = 2'732,736
                inputDim = self.cnn['conv6']['output']

            if convFilters == 3:
                self.cnn = {'conv1':{'filters':64,  'size':5, 'stride':2, 'output':(112**2)*64 }, # Feature map size 802,816 - Params: 5x5x3x64    =     4,800
                            'conv2':{'filters':128, 'size':3, 'stride':2, 'output':(56**2)*128 }, # Feature map size 401,408 - Params: 3x3x64x128  =    73,728
                            'conv3':{'filters':64,  'size':3, 'stride':1, 'output':(56**2)*64  }, # Feature map size 200,704 - Params: 3x3x128x64  =    73,728
                            'conv4':{'filters':32,  'size':3, 'stride':1, 'output':(56**2)*32  }, # Feature map size 100,352 - Params: 3x3x64x32   =    18,432
                            'conv5':{'filters':16,  'size':3, 'stride':1, 'output':(56**2)*16  }, # Feature map size  50,176 - Params: 3x3x32x16   =     4,608
                            'conv6':{'filters': 8,  'size':3, 'stride':1, 'output':(56**2)*8   }, # Feature map size  25,088 - Params: 3x3x16x8    =     1,152
                            'pad':'half'}                                                         #        TOTALS: 1'580,540 -                     =   176,448
                inputDim = self.cnn['conv6']['output']
            if convFilters == 4:
                self.cnn = {'conv1':{'filters':96, 'size':7, 'stride':2, 'output':(112**2)*96 },  # Feature map size 1'204,224 - Params: 7x7x3x96    =    14,112
                            'conv2':{'filters':128, 'size':5, 'stride':2, 'output':(56**2)*128 }, # Feature map size   401,408 - Params: 5x5x96x128  =   307,200
                            'conv3':{'filters':256, 'size':3, 'stride':2, 'output':(28**2)*256 }, # Feature map size   200,704 - Params: 3x3x128x256 =   294,912
                            'conv4':{'filters':256, 'size':3, 'stride':2, 'output':(14**2)*256 }, # Feature map size   100,352 - Params: 3x3x256x256 =   589,824
                            'conv5':{'filters':512, 'size':3, 'stride':2, 'output':(7**2)*512 },  # Feature map size    25,088 - Params: 3x3x256x512 = 1'179,648
                            'conv6':{'filters':256, 'size':3, 'stride':1, 'output':(7**2)*256 },  # Feature map size    12,544 - Params: 3x3x512x256 = 1'179,648
                            'pad':'half'}                                                         #          TOTALS: 1'944,320 -                     = 3'565,344
                inputDim = self.cnn['conv6']['output']

        self.targetDim = targetDim
        self.inputDim = inputDim #+ self.targetDim
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.norm = norm
        self.stateDim = [stateDim, stateDim]
        self.imgSize = imgSize
        self.useAttention = useAttention
        self.computeFlow = computeFlow
        if self.computeFlow:
            if self.modelArch.endswith('ConvLayers'):
                self.channels = 5
            else:
                print 'Flow not supported for these models'
                self.channels = 3
                self.computeFlow = False
        else:
            self.channels = 3
        if self.useAttention == 'squareChannel':
            self.channels += 1
        self.buildModel(self.batchSize, self.inputDim, self.stateDim, self.targetDim, zeroTailFc, learningRate, use_cudnn, self.imgSize)

    def preprocess(self, data, label, flow):
        # Adjust channels and normalize pixels
        if self.modelArch.endswith('ConvLayers'):
            data = (data - 127.)/127.
            if self.computeFlow: data = NP.append(data, flow, axis=4)
            data = NP.swapaxes(NP.swapaxes(data, 3, 4), 2, 3)
        elif self.modelArch == 'lasagne':
            data = self.cnn.prepareBatch(data)
        # Prepare attention masks on first frame
        if self.useAttention == 'square':
            firstFrameMasks = VisualAttention.getSquaredMasks(data[:,0,...], label[:,0,:])
            data[:,0,...] *= firstFrameMasks #Multiplicative mask
        elif self.useAttention == 'squareChannel':
            b,t,c,w,h = data.shape
            firstFrameMasks = VisualAttention.getSquaredMaskChannel(data[:,0,...], label[:,0,:])
            data = NP.append(data, NP.zeros((b,t,1,w,h)), axis=2)
            data[:,0,-1,:,:] = firstFrameMasks
        # Center labels around zero, and scale them between [-1,1]
        label = label / (self.imgSize / 2.) - 1.
        return data, label
    
    def fit(self, data, label, flow):
        data, label = self.preprocess(data, label, flow)
        return self.fitFunc(self.seqLength, data, label[:, 0, :], label)
      
    def forward(self, data, label, flow):
        data, label = self.preprocess(data, label, flow)
        cost, output = self.forwardFunc(self.seqLength, data, label[:, 0, :], label)
        return cost, output
    
    def buildModel(self, batchSize, inputDim, stateDim, targetDim, zeroTailFc, initialLearningRate, use_cudnn, imgSize):
        logging.info('Building network')
        
        # imgs: of shape (batchSize, seq_len, nr_channels, img_rows, img_cols)
        imgs = getTensor("images", Theano.config.floatX, 5)
        starts = Tensor.matrix()
        
        #Select conv2d implementation
        conv2d = initializeConv2d(use_cudnn)

        ## Attention mask
        attention = VisualAttention.buildAttention(self.useAttention, imgSize)

        params = list(self.init_params(inputDim, stateDim, targetDim, zeroTailFc))
        if self.modelArch.endswith('ConvLayers'):
            Wr1, Ur1, br1, Wz1, Uz1, bz1, Wg1, Ug1, bg1, Wr2, Ur2, br2, Wz2, Uz2, bz2, Wg2, Ug2, bg2, Wfc3, bfc3 = params[:20]
            convParams = params[20:]
            def step(img, prev_bbox, state1, state2):
                img = attention(img, prev_bbox)
                features = buildCNN(img, self.cnn, convParams)
                h1 = gru(features, prev_bbox, state1, Wr1, Ur1, br1, Wz1, Uz1, bz1, Wg1, Ug1, bg1)
                h2 = gru(h1, prev_bbox, state2, Wr2, Ur2, br2, Wz2, Uz2, bz2, Wg2, Ug2, bg2)
                boxes = boxRegressor(h2, Wfc3, bfc3)
                return boxes, h1, h2
        elif self.modelArch == 'caffe':
            Wr1, Ur1, br1, Wz1, Uz1, bz1, Wg1, Ug1, bg1, Wr2, Ur2, br2, Wz2, Uz2, bz2, Wg2, Ug2, bg2, Wfc3, bfc3 = params
            def step(img, prev_bbox, state1, state2):
                features = img
                h1 = gru(features, prev_bbox, state1, Wr1, Ur1, br1, Wz1, Uz1, bz1, Wg1, Ug1, bg1)
                h2 = gru(h1, prev_bbox, state2, Wr2, Ur2, br2, Wz2, Uz2, bz2, Wg2, Ug2, bg2)
                boxes = boxRegressor(h2, Wfc3, bfc3)
                return boxes, h1, h2
        elif self.modelArch == 'lasagne':
            Wr1, Ur1, br1, Wz1, Uz1, bz1, Wg1, Ug1, bg1, Wr2, Ur2, br2, Wz2, Uz2, bz2, Wg2, Ug2, bg2, Wfc3, bfc3 = params
            def step(img, prev_bbox, state1, state2):
                img = attention(img, prev_bbox)
                features = self.cnn.getFeatureExtractor(img)
                h1 = gru(features, prev_bbox, state1, Wr1, Ur1, br1, Wz1, Uz1, bz1, Wg1, Ug1, bg1)
                h2 = gru(h1, prev_bbox, state2, Wr2, Ur2, br2, Wz2, Uz2, bz2, Wg2, Ug2, bg2)
                boxes = boxRegressor(h2, Wfc3, bfc3)
                return boxes, h1, h2
               
        state1 = Tensor.zeros((batchSize, stateDim[0]))
        state2 = Tensor.zeros((batchSize, stateDim[1]))
        # Move the time axis to the top
        sc, _ = Theano.scan(step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, state1, state2])
    
        bbox_seq = sc[0].dimshuffle(1, 0, 2)
    
        # targets: of shape (batch_size, seq_len, targetDim)
        targets = getTensor("targets", Theano.config.floatX, 3)
        seq_len_scalar = Tensor.scalar()
    
        cost = self.norm(targets - bbox_seq).sum() / batchSize / seq_len_scalar

        # Learning rate
        learning_rate_decay = 0.95
        learningRate = Theano.shared(NP.asarray(initialLearningRate, dtype=Theano.config.floatX))
        decayLearningRate = Theano.function(inputs=[], outputs=learningRate, updates={learningRate: learningRate * learning_rate_decay})
    
        logging.info('Building optimizer')
    
        fitFunc = Theano.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], updates=rmsprop(cost, params, learningRate), allow_input_downcast=True)
        forwardFunc = Theano.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], allow_input_downcast=True)
        imgStep = getTensor("images", Theano.config.floatX, 4)
        startsStep = Tensor.matrix()
        state1Step = Tensor.matrix()
        state2Step = Tensor.matrix()
        stepFunc = Theano.function([imgStep, startsStep, state1Step, state2Step], step(imgStep, startsStep, state1Step, state2Step))

        self.learningRate = learningRate
        self.decayLearningRate = decayLearningRate
        self.fitFunc, self.forwardFunc, self.params, self.stepFunc = fitFunc, forwardFunc, params, stepFunc
    
    
    def init_params(self, inputDim, stateDim, targetDim, zeroTailFc):
        ### NETWORK PARAMETERS BEGIN
        convParams = initCNN(self.cnn, self.channels)
        gru1 = initGru(inputDim, stateDim[0], '1')
        gru2 = initGru(stateDim[0], stateDim[1], '2')
        regressor = initRegressor(stateDim[1], targetDim, zeroTailFc)
        ### NETWORK PARAMETERS END
    
        if self.modelArch.endswith('ConvLayers'):
            return tuple(gru1) + tuple(gru2) + tuple(regressor) + tuple(convParams)
        else:
            return tuple(gru1) + tuple(gru2) + regressor
