import theano as Theano
import theano.tensor as Tensor
import numpy as NP
import numpy.random as RNG
import theano.tensor.nnet as NN
#TODO: pickle or cPickle?
import pickle

from collections import OrderedDict



class TheanoGruRnn(object):
    
    fitFunc = None
    forwardFunc = None
    params = None
    seqLength = None
    
    def __init__(self, inputDim, stateDim, batchSize, seqLength, zeroTailFc, learningRate, use_cudnn, imgSize, pretrained=False):
        ### Computed hyperparameters begin
        self.pretrained = pretrained
        if not self.pretrained:
            #Number of feature filters
            self.conv_nr_filters = 32
            #Rows/cols of feature filters
            self.conv_filter_row = self.conv_filter_col = 10
            self.conv_stride = 5
            #TODO: pass image dims
            inputDim = ((imgSize - self.conv_filter_row) / self.conv_stride + 1) * \
                        ((imgSize - self.conv_filter_col) / self.conv_stride + 1) * \
                        self.conv_nr_filters
        self.inputDim = inputDim + 4
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.fitFunc, self.forwardFunc, self.params = self.buildModel(self.batchSize, self.inputDim, stateDim, zeroTailFc, learningRate, use_cudnn)

    
    def fit(self, data, label):
        return self.fitFunc(self.seqLength, data, label[:, 0, :], label)
      
        
    def forward(self, data, label):
        cost, output = self.forwardFunc(self.seqLength, data, label[:, 0, :], label)
        return output
    
    
    def loadModel(self, modelPath):
        try:
            print 'Loading model from {}'.format(modelPath)
            f = open(modelPath, "rb")
            param_saved = pickle.load(f)
            for _p, p in zip(self.params, param_saved):
                _p.set_value(p)
        except Exception as e:
            print 'Exception loading model {}: {}'.format(modelPath, e)
      
    def saveModel(self, modelPath):
        with open(modelPath, 'wb') as trackerFile:
            #TODO: verify the protocol allows sharing of models within users
            pickle.dump(self.params, trackerFile, pickle.HIGHEST_PROTOCOL)
        
    def getTensor(self, name, dtype, dim):
        if dtype == None:
            dtype = Theano.config.floatX
        
        return Tensor.TensorType(dtype, [False] * dim, name=name)()
        
    
    def buildModel(self, batchSize, inputDim, stateDim, zeroTailFc, learningRate, use_cudnn):
        print 'Building network'
        
        # imgs: of shape (batchSize, seq_len, nr_channels, img_rows, img_cols)
        imgs = self.getTensor("images", Theano.config.floatX, 5)
        starts = Tensor.matrix()
        
        conv2d = NN.conv2d
        if use_cudnn and Theano.config.device[:3] == 'gpu':
            import theano.sandbox.cuda.dnn as CUDNN
            if CUDNN.dnn_available():
                print 'Using CUDNN instead of Theano conv2d'
                conv2d = CUDNN.dnn_conv

        params = list(self.init_params(inputDim, stateDim, zeroTailFc))
        if not self.pretrained:
            conv_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                # of (batch_size, nr_filters, some_rows, some_cols)
                conv1 = conv2d(img, conv_filters, subsample=(self.conv_stride, self.conv_stride))
                act1 = Tensor.tanh(conv1)
                flat1 = Tensor.reshape(act1, (batchSize, inputDim-4))
                gru_in = Tensor.concatenate([flat1, prev_bbox], axis=1)
                gru_z = NN.sigmoid(Tensor.dot(gru_in, Wz) + Tensor.dot(state, Uz) + bz)
                gru_r = NN.sigmoid(Tensor.dot(gru_in, Wr) + Tensor.dot(state, Ur) + br)
                gru_h_ = Tensor.tanh(Tensor.dot(gru_in, Wg) + Tensor.dot(gru_r * state, Ug) + bg)
                gru_h = (1-gru_z) * state + gru_z * gru_h_
                bbox = Tensor.tanh(Tensor.dot(gru_h, W_fc2) + b_fc2)
                return bbox, gru_h
        else:
            Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = params
            def step(img, prev_bbox, state):
                # of (batch_size, nr_filters, some_rows, some_cols)
                act1 = img
                flat1 = Tensor.reshape(act1, (batchSize, inputDim-4))
                gru_in = Tensor.concatenate([flat1, prev_bbox], axis=1)
                gru_z = NN.sigmoid(Tensor.dot(gru_in, Wz) + Tensor.dot(state, Uz) + bz)
                gru_r = NN.sigmoid(Tensor.dot(gru_in, Wr) + Tensor.dot(state, Ur) + br)
                gru_h_ = Tensor.tanh(Tensor.dot(gru_in, Wg) + Tensor.dot(gru_r * state, Ug) + bg)
                gru_h = (1-gru_z) * state + gru_z * gru_h_
                bbox = Tensor.tanh(Tensor.dot(gru_h, W_fc2) + b_fc2)
                return bbox, gru_h
            
               
        # Move the time axis to the top
        sc, _ = Theano.scan(step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, Tensor.zeros((batchSize, stateDim))])
    
        bbox_seq = sc[0].dimshuffle(1, 0, 2)
    
        # targets: of shape (batch_size, seq_len, 4)
        targets = self.getTensor("targets", Theano.config.floatX, 3)
        seq_len_scalar = Tensor.scalar()
    
        cost = ((targets - bbox_seq) ** 2).sum() / batchSize / seq_len_scalar
    
        print 'Building optimizer'
    
        fitFunc = Theano.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], updates=self.rmsprop(cost, params, learningRate), allow_input_downcast=True)
        forwardFunc = Theano.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], allow_input_downcast=True)
        
        return fitFunc, forwardFunc, params
    
    
    def init_params(self, inputDim, stateDim, zeroTailFc):
        ### NETWORK PARAMETERS BEGIN
        if not self.pretrained:
            conv_filters = Theano.shared(self.glorot_uniform((self.conv_nr_filters, 1, self.conv_filter_row, self.conv_filter_col)), name='conv_filters')
        Wr = Theano.shared(self.glorot_uniform((inputDim, stateDim)), name='Wr')
        Ur = Theano.shared(self.orthogonal((stateDim, stateDim)), name='Ur')
        br = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='br')
        Wz = Theano.shared(self.glorot_uniform((inputDim, stateDim)), name='Wz')
        Uz = Theano.shared(self.orthogonal((stateDim, stateDim)), name='Uz')
        bz = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='bz')
        Wg = Theano.shared(self.glorot_uniform((inputDim, stateDim)), name='Wg')
        Ug = Theano.shared(self.orthogonal((stateDim, stateDim)), name='Ug')
        bg = Theano.shared(NP.zeros((stateDim,), dtype=Theano.config.floatX), name='bg')
        W_fc2 = Theano.shared(self.glorot_uniform((stateDim, 4)) if not zeroTailFc else NP.zeros((stateDim, 4), dtype=Theano.config.floatX), name='W_fc2')
        b_fc2 = Theano.shared(NP.zeros((4,), dtype=Theano.config.floatX), name='b_fc2')
        ### NETWORK PARAMETERS END
    
        if not self.pretrained:
            return conv_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2
        else:
            return Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2
    
    
    def rmsprop(self, cost, params, lr=0.0005, rho=0.9, epsilon=1e-6):
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
    
    
    def glorot_uniform(self, shape):
        '''
        Borrowed from keras
        '''
        fan_in, fan_out = self.get_fans(shape)
        s = NP.sqrt(6. / (fan_in + fan_out))
        return NP.cast[Theano.config.floatX](RNG.uniform(low=-s, high=s, size=shape))
    
    
    def get_fans(self, shape):
        '''
        Borrowed from keras
        '''
        fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    
    
    def orthogonal(self, shape, scale=1.1):
        '''
        Borrowed from keras
        '''
        flat_shape = (shape[0], NP.prod(shape[1:]))
        a = RNG.normal(0, 1, flat_shape)
        u, _, v = NP.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        
        return NP.cast[Theano.config.floatX](q)
