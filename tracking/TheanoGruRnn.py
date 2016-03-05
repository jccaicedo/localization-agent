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
def gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2):
    flat1 = Tensor.reshape(features, (features.shape[0], Tensor.prod(features.shape[1:])))
    gru_in = Tensor.concatenate([flat1, prev_bbox], axis=1)
    gru_z = NN.sigmoid(Tensor.dot(gru_in, Wz) + Tensor.dot(state, Uz) + bz)
    gru_r = NN.sigmoid(Tensor.dot(gru_in, Wr) + Tensor.dot(state, Ur) + br)
    gru_h_ = Tensor.tanh(Tensor.dot(gru_in, Wg) + Tensor.dot(gru_r * state, Ug) + bg)
    gru_h = (1-gru_z) * state + gru_z * gru_h_
    bbox = Tensor.tanh(Tensor.dot(gru_h, W_fc2) + b_fc2)
    return bbox, gru_h
    
def initGru(inputDim, stateDim, targetDim, zeroTailFc):
    Wr = Theano.shared(glorot_uniform((inputDim, stateDim)), name='Wr')
    Ur = Theano.shared(orthogonal((stateDim, stateDim)), name='Ur')
    br = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='br')
    Wz = Theano.shared(glorot_uniform((inputDim, stateDim)), name='Wz')
    Uz = Theano.shared(orthogonal((stateDim, stateDim)), name='Uz')
    bz = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='bz')
    Wg = Theano.shared(glorot_uniform((inputDim, stateDim)), name='Wg')
    Ug = Theano.shared(orthogonal((stateDim, stateDim)), name='Ug')
    bg = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='bg')
    if not zeroTailFc:
        W_fc2init = glorot_uniform((stateDim, targetDim))
    else:
        W_fc2init = NP.zeros((stateDim, targetDim), dtype=Theano.config.floatX)
    W_fc2 = Theano.shared(W_fc2init, name='W_fc2')
    b_fc2 = Theano.shared(NP.zeros((targetDim,), dtype=Theano.config.floatX), name='b_fc2')
    return Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2

def initializeConv2d(use_cudnn=False):
    conv2d = NN.conv2d
    if use_cudnn and Theano.config.device[:3] == 'gpu':
        import theano.sandbox.cuda.dnn as CUDNN
        if CUDNN.dnn_available():
            logging.warning('Using CUDNN instead of Theano conv2d')
            conv2d = CUDNN.dnn_conv
    return conv2d

def buildAttention(useAttention, imgSize):
    if useAttention == 'gaussian':
        attention = VisualAttention.createGaussianMasker(imgSize)
    elif useAttention == 'square':
        attention = VisualAttention.createSquareMasker(imgSize)
    else:
        attention = VisualAttention.useNoMask()
    return attention

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
    
    def __init__(self, inputDim, stateDim, targetDim, batchSize, seqLength, zeroTailFc, learningRate, use_cudnn, imgSize, modelArch='base', norm=l2, useAttention=False, modelPath=None, layerKey=None, convFilters=32):
        ### Computed hyperparameters begin
        self.modelArch = modelArch
        if self.modelArch == 'base':
            #TODO: change to same structure as multiple conv layers
            #Number of feature filters
            self.conv_nr_filters = convFilters
            #Rows/cols of feature filters
            self.conv_filter_row = self.conv_filter_col = 10
            self.conv_stride = 5
            #TODO: pass image dims
            inputDim = ((imgSize - self.conv_filter_row) / self.conv_stride + 1) * \
                        ((imgSize - self.conv_filter_col) / self.conv_stride + 1) * \
                        self.conv_nr_filters
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
                            'conv3':{'filters':16, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*16 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 2:
                self.cnn = {'conv1':{'filters':32, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*32 },
                            'conv2':{'filters':32, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*32 },
                            'conv3':{'filters':32, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*32 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 3:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*64 },
                            'conv2':{'filters':32, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*32 },
                            'conv3':{'filters':32, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*32 }}
                inputDim = self.cnn['conv3']['output']
            if convFilters == 4:
                self.cnn = {'conv1':{'filters':64, 'size':5, 'stride':2, 'output':(((192-5)/2+1)**2)*64 },
                            'conv2':{'filters':64, 'size':3, 'stride':2, 'output':(((94-3)/2+1)**2)*64 },
                            'conv3':{'filters':32, 'size':3, 'stride':2, 'output':(((46-3)/2+1)**2)*64 }}
                inputDim = self.cnn['conv3']['output']

        self.targetDim = targetDim
        self.inputDim = inputDim + self.targetDim
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.norm = norm
        self.stateDim = stateDim
        self.imgSize = imgSize
        self.useAttention = useAttention
        self.fitFunc, self.forwardFunc, self.params, self.stepFunc = self.buildModel(self.batchSize, self.inputDim, self.stateDim, self.targetDim, zeroTailFc, learningRate, use_cudnn, self.imgSize, self.useAttention)

    
    def fit(self, data, label):
        if self.modelArch == 'lasagne':
            data = self.cnn.prepareBatch(data)
        elif self.modelArch.endswith('ConvLayers'):
            data = NP.swapaxes(NP.swapaxes(data, 3, 4), 2, 3)
            data = (data - 127.)/127.
        return self.fitFunc(self.seqLength, data, label[:, 0, :], label)
      
        
    def forward(self, data, label):
        if self.modelArch == 'lasagne':
          data = self.cnn.prepareBatch(data)
        elif self.modelArch.endswith('ConvLayers'):
            data = NP.swapaxes(NP.swapaxes(data, 3, 4), 2, 3)
            data = (data - 127.)/127.
        cost, output = self.forwardFunc(self.seqLength, data, label[:, 0, :], label)
        return cost, output
    
    def buildModel(self, batchSize, inputDim, stateDim, targetDim, zeroTailFc, learningRate, use_cudnn, imgSize, useAttention):
        logging.info('Building network')
        
        # imgs: of shape (batchSize, seq_len, nr_channels, img_rows, img_cols)
        imgs = getTensor("images", Theano.config.floatX, 5)
        starts = Tensor.matrix()
        
        #Select conv2d implementation
        conv2d = initializeConv2d(use_cudnn)

        ## Attention mask
        attention = buildAttention(useAttention, imgSize)

        params = list(self.init_params(inputDim, stateDim, targetDim, zeroTailFc))
        if self.modelArch == 'base':
            conv1, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                img = attention(img, prev_bbox)
                # of (batch_size, nr_filters, some_rows, some_cols)
                fmap1 = conv2d(img, conv1, subsample=(self.conv_stride, self.conv_stride))
                #TODO: compare both functions                
                act1 = Tensor.tanh(fmap1)
                #act1 = Tensor.nnet.relu(fmap1)
                features = act1
                return gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2)
        elif self.modelArch == 'caffe':
            Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                # of (batch_size, nr_filters, some_rows, some_cols)
                act1 = img
                features = act1
                return gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2)
        elif self.modelArch == 'lasagne':
            Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                img = attention(img, prev_bbox)
                features = self.cnn.getFeatureExtractor(img)
                return gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2)
        elif self.modelArch == 'twoConvLayers':
            conv1, conv2, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                img = attention(img, prev_bbox)
                fmap1 = conv2d(img, conv1, subsample=(self.cnn['conv1']['stride'], self.cnn['conv1']['stride']))
                act1 = Tensor.nnet.relu(fmap1)
                fmap2 = conv2d(act1, conv2, subsample=(self.cnn['conv2']['stride'], self.cnn['conv2']['stride']))
                act2 = Tensor.nnet.relu(fmap2)
                features = act2
                return gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2)
        elif self.modelArch == 'threeConvLayers':
            conv1, conv2, conv3, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                img = attention(img, prev_bbox)
                fmap1 = conv2d(img, conv1, subsample=(self.cnn['conv1']['stride'], self.cnn['conv1']['stride']))
                act1 = Tensor.nnet.relu(fmap1)
                fmap2 = conv2d(act1, conv2, subsample=(self.cnn['conv2']['stride'], self.cnn['conv2']['stride']))
                act2 = Tensor.nnet.relu(fmap2)
                fmap3 = conv2d(act2, conv3, subsample=(self.cnn['conv3']['stride'], self.cnn['conv3']['stride']))
                act3 = Tensor.nnet.relu(fmap3)
                features = act3
                return gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2)
               
        state = Tensor.zeros((batchSize, stateDim))
        # Move the time axis to the top
        sc, _ = Theano.scan(step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, state])
    
        bbox_seq = sc[0].dimshuffle(1, 0, 2)
    
        # targets: of shape (batch_size, seq_len, targetDim)
        targets = getTensor("targets", Theano.config.floatX, 3)
        seq_len_scalar = Tensor.scalar()
    
        cost = self.norm(targets - bbox_seq).sum() / batchSize / seq_len_scalar
    
        logging.info('Building optimizer')
    
        fitFunc = Theano.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], updates=rmsprop(cost, params, learningRate), allow_input_downcast=True)
        forwardFunc = Theano.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], allow_input_downcast=True)
        imgStep = getTensor("images", Theano.config.floatX, 4)
        startsStep = Tensor.matrix()
        stateStep = Tensor.matrix()
        stepFunc = Theano.function([imgStep, startsStep, stateStep], step(imgStep, startsStep, stateStep))
        
        return fitFunc, forwardFunc, params, stepFunc
    
    
    def init_params(self, inputDim, stateDim, targetDim, zeroTailFc):
        ### NETWORK PARAMETERS BEGIN
        if self.modelArch == 'base':
            conv1 = Theano.shared(glorot_uniform((self.conv_nr_filters, 1, self.conv_filter_row, self.conv_filter_col)), name='conv1')
        if self.modelArch == 'twoConvLayers':
            channels = 3
            conv1 = Theano.shared(self.glorot_uniform((self.cnn['conv1']['filters'], channels, self.cnn['conv1']['size'], self.cnn['conv1']['size'])), name='conv1')
            conv2 = Theano.shared(self.glorot_uniform((self.cnn['conv2']['filters'], self.cnn['conv1']['filters'], self.cnn['conv2']['size'], self.cnn['conv2']['size'])), name='conv2')
        if self.modelArch == 'threeConvLayers':
            channels = 3
            conv1 = Theano.shared(self.glorot_uniform((self.cnn['conv1']['filters'], channels, self.cnn['conv1']['size'], self.cnn['conv1']['size'])), name='conv1')
            conv2 = Theano.shared(self.glorot_uniform((self.cnn['conv2']['filters'], self.cnn['conv1']['filters'], self.cnn['conv2']['size'], self.cnn['conv2']['size'])), name='conv2')
            conv3 = Theano.shared(self.glorot_uniform((self.cnn['conv3']['filters'], self.cnn['conv2']['filters'], self.cnn['conv3']['size'], self.cnn['conv3']['size'])), name='conv3')
        Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = initGru(inputDim, stateDim, targetDim, zeroTailFc)
        ### NETWORK PARAMETERS END
    
        if self.modelArch == 'base':
            return conv1, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2
        if self.modelArch == 'twoConvLayers':
            return conv1, conv2, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2
        if self.modelArch == 'threeConvLayers':
            return conv1, conv2, conv3, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2
        else:
            return Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2
