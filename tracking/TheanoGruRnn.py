import theano as Theano
import theano.tensor as Tensor
import numpy as NP
import numpy.random as RNG
import theano.tensor.nnet as NN
import cPickle as pickle
import VisualAttention
import logging

from collections import OrderedDict
from TheanoConvNet import TheanoConvNet

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
    flat1 = Tensor.reshape(features, (features.shape[0], Tensor.prod(features.shape[1:])))
    gru_in = Tensor.concatenate([flat1, prev_bbox], axis=1) #TODO: Remove this thing!
    gru_z = NN.sigmoid(Tensor.dot(gru_in, Wz) + Tensor.dot(state, Uz) + bz)
    gru_r = NN.sigmoid(Tensor.dot(gru_in, Wr) + Tensor.dot(state, Ur) + br)
    gru_h_ = Tensor.tanh(Tensor.dot(gru_in, Wg) + Tensor.dot(gru_r * state, Ug) + bg)
    gru_h = (1-gru_z) * state + gru_z * gru_h_
    return gru_h

def boxRegressor(gru_h, W_fc, b_fc):
    bbox = Tensor.tanh(Tensor.dot(gru_h, W_fc) + b_fc)
    return bbox, gru_h
    
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
    return model
      
def saveModel(model, modelPath):
    #TODO: silent for trax
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
        if self.modelArch.endswith('ConvLayers'):
            self.cnn = TheanoConvNet(modelArch, convFilters, use_cudnn)
            inputDim = self.cnn.output
        elif self.modelArch == 'lasagne':
            from LasagneVGG16 import LasagneVGG16
            self.cnn = LasagneVGG16(modelPath, layerKey)
            inputDim = 512 * 7 * 7

        self.targetDim = targetDim
        self.inputDim = inputDim + self.targetDim
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.norm = norm
        self.stateDim = stateDim
        self.imgSize = imgSize
        self.useAttention = useAttention
        self.computeFlow = computeFlow
        if self.computeFlow:
            if self.modelArch.endswith('ConvLayers'):
                self.channels = 5
                logging.info('Computing optic flow')
            else:
                logging.warning('Flow not supported for these models')
                self.channels = 3
                self.computeFlow = False
        else:
            self.channels = 3
        if self.useAttention == 'squareChannel':
            self.channels += 1
            logging.info('Adding extra mask channel')
        self.buildModel(self.batchSize, self.inputDim, self.stateDim, self.targetDim, zeroTailFc, learningRate, use_cudnn, self.imgSize)

    def preprocess(self, data, label, flow):
        # Adjust channels and normalize pixels
        if self.modelArch.endswith('ConvLayers'):
            if self.computeFlow: data = NP.append(data, flow, axis=4)
            data = (data - 127.)/127.
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
        # Normalize labels according to the chosen mask
        label = VisualAttention.stdLabels(label, self.imgSize)
        return data, label
    
    #We could be interested in data post processing but for now only on labels
    def postprocess(self, label):
        label = VisualAttention.stdBoxes(label, self.imgSize)
        return label
    
    def postprocessData(self, data):
        # Adjust channels and normalize pixels
        if self.modelArch.endswith('ConvLayers'):
            data = NP.swapaxes(NP.swapaxes(data, 2, 3), 3, 4)
            #Unconditional 
            data = data[...,:3]
            data = (data * 127.) + 127.
        elif self.modelArch == 'lasagne':
            raise Exception('Not implemented yet')
        return data
    
    def fit(self, data, label, flow):
        data, label = self.preprocess(data, label, flow)
        cost, bbox = self.fitFunc(self.seqLength, data, label[:, 0, :], label)
        return cost, self.postprocess(bbox)
      
    def forward(self, data, label, flow):
        data, label = self.preprocess(data, label, flow)
        cost, bbox = self.forwardFunc(self.seqLength, data, label[:, 0, :], label)
        return cost, self.postprocess(bbox)
    
    def buildModel(self, batchSize, inputDim, stateDim, targetDim, zeroTailFc, initialLearningRate, use_cudnn, imgSize):
        logging.info('Building network')
        
        # imgs: of shape (batchSize, seq_len, nr_channels, img_rows, img_cols)
        imgs = getTensor("images", Theano.config.floatX, 5)
        starts = Tensor.matrix()
        
        ## Attention mask
        attention = VisualAttention.buildAttention(self.useAttention, imgSize)

        params = list(self.init_params(inputDim, stateDim, targetDim, zeroTailFc))
        if self.modelArch.endswith('ConvLayers'):
            Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params[:11]
            convParams = params[11:]
            def step(img, prev_bbox, state):
                img = attention(img, prev_bbox)
                features = self.cnn.buildCNN(img, convParams)
                return boxRegressor( gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg), W_fc2, b_fc2)
        elif self.modelArch == 'caffe':
            Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                # of (batch_size, nr_filters, some_rows, some_cols)
                act1 = img
                features = act1
                return boxRegressor( gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg), W_fc2, b_fc2)
        elif self.modelArch == 'lasagne':
            Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                img = attention(img, prev_bbox)
                features = self.cnn.getFeatureExtractor(img)
                return boxRegressor( gru(features, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg), W_fc2, b_fc2)
               
        state = Tensor.zeros((batchSize, stateDim))
        # Move the time axis to the top
        sc, _ = Theano.scan(step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, state])
    
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
        stateStep = Tensor.matrix()
        stepFunc = Theano.function([imgStep, startsStep, stateStep], step(imgStep, startsStep, stateStep))
        
        self.learningRate = learningRate
        self.decayLearningRate = decayLearningRate
        self.fitFunc, self.forwardFunc, self.params, self.stepFunc = fitFunc, forwardFunc, params, stepFunc
    
    
    def init_params(self, inputDim, stateDim, targetDim, zeroTailFc):
        ### NETWORK PARAMETERS BEGIN
        Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg = initGru(inputDim, stateDim, '1')
        W_fc2, b_fc2 = initRegressor(stateDim, targetDim, zeroTailFc)
        ### NETWORK PARAMETERS END
    
        if self.modelArch.endswith('ConvLayers'):
            convParams = self.cnn.initCNN(self.channels)
            return (Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2) + tuple(convParams)
        else:
            return (Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2)
