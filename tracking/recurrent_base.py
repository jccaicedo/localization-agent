import argparse as ap

import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG

import numpy as NP
import numpy.random as RNG
import GaussianGenerator as GG

from collections import OrderedDict

import cPickle
import time

def clock(m, st): 
  print m,(time.time()-st)

#####################################################################
# Usage:                                                            #
# python -u recurrent_plain_base.py [opts] [model_name]             #
#                                                                   #
# Options:                                                          #
#       --batch_size=INTEGER                                        #
#       --conv1_nr_filters=INTEGER                                  #
#       --conv1_filter_size=INTEGER                                 #
#       --conv1_stride=INTEGER                                      #
#       --img_size=INTEGER                                          #
#       --gru_dim=INTEGER                                           #
#       --seq_len=INTEGER                                           #
#       --use_cudnn     (Set floatX to float32 if you use this)     #
#       --zero_tail_fc  (Recommended)                               #
#####################################################################

### Utility functions begin
def get_fans(shape):
	'''
	Borrowed from keras
	'''
	fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
	fan_out = shape[1] if len(shape) == 2 else shape[0]
	return fan_in, fan_out

def glorot_uniform(shape):
	'''
	Borrowed from keras
	'''
	fan_in, fan_out = get_fans(shape)
	s = NP.sqrt(6. / (fan_in + fan_out))
	return NP.cast[T.config.floatX](RNG.uniform(low=-s, high=s, size=shape))

def orthogonal(shape, scale=1.1):
	'''
	Borrowed from keras
	'''
	flat_shape = (shape[0], NP.prod(shape[1:]))
	a = RNG.normal(0, 1, flat_shape)
	u, _, v = NP.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return NP.cast[T.config.floatX](q)

def tensor5(name=None, dtype=None):
	if dtype == None:
		dtype = T.config.floatX
	return TT.TensorType(dtype, [False] * 5, name=name)()

### RMSProp begin
def rmsprop(cost, params, lr=0.0005, rho=0.9, epsilon=1e-6):
	'''
	Borrowed from keras, no constraints, though
	'''
	updates = OrderedDict()
	grads = T.grad(cost, params)
	acc = [T.shared(NP.zeros(p.get_value().shape, dtype=T.config.floatX)) for p in params]
	for p, g, a in zip(params, grads, acc):
		new_a = rho * a + (1 - rho) * g ** 2
		updates[a] = new_a
		new_p = p - lr * g / TT.sqrt(new_a + epsilon)
		updates[p] = new_p

	return updates
### RMSprop end

def dump_params(model_name, params):
    f = open(model_name, "wb")
    cPickle.dump(map(lambda x: x.get_value(), params), f)
    f.close()

def build(batch_size, gru_input_dim, gru_dim, conv_output_dim, conv_nr_filters, conv_filter_row, conv_filter_col, conv_stride, zero_tail_fc, test, use_cudnn, learning_rate):
    print 'Building network'

    # imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
    imgs = tensor5()
    starts = TT.matrix()

    conv2d = NN.conv2d
    if use_cudnn and T.config.device[:3] == 'gpu':
        import theano.sandbox.cuda.dnn as CUDNN
        if CUDNN.dnn_available():
            print 'Using CUDNN instead of Theano conv2d'
            conv2d = CUDNN.dnn_conv

    conv_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = init_params(conv_nr_filters, conv_filter_row, conv_filter_col, gru_input_dim, gru_dim, zero_tail_fc)
    params = [conv_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2]

    ### Recurrent step
    # img: of shape (batch_size, nr_channels, img_rows, img_cols)
    def _step(img, prev_bbox, state):
        # of (batch_size, nr_filters, some_rows, some_cols)
        conv1 = conv2d(img, conv_filters, subsample=(conv_stride, conv_stride))
        act1 = TT.tanh(conv1)
        flat1 = TT.reshape(act1, (batch_size, conv_output_dim))
        gru_in = TT.concatenate([flat1, prev_bbox], axis=1)
        gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
        gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
        gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
        gru_h = (1-gru_z) * state + gru_z * gru_h_
        bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)
        return bbox, gru_h

    # Move the time axis to the top
    sc, _ = T.scan(_step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, TT.zeros((batch_size, gru_dim))])

    bbox_seq = sc[0].dimshuffle(1, 0, 2)

    # targets: of shape (batch_size, seq_len, 4)
    targets = TT.tensor3()
    seq_len_scalar = TT.scalar()

    cost = ((targets - bbox_seq) ** 2).sum() / batch_size / seq_len_scalar

    print 'Building optimizer'

    train = T.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], updates=rmsprop(cost, params, lr=learning_rate) if not test else None, allow_input_downcast=True)
    tester = T.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], allow_input_downcast=True)
    
    return train, tester, params

def init_params(conv_nr_filters, conv_filter_row, conv_filter_col, gru_input_dim, gru_dim, zero_tail_fc):
    
    print 'Initializing parameters'

    ### NETWORK PARAMETERS BEGIN
    conv_filters = T.shared(glorot_uniform((conv_nr_filters, 1, conv_filter_row, conv_filter_col)), name='conv_filters')
    Wr = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wr')
    Ur = T.shared(orthogonal((gru_dim, gru_dim)), name='Ur')
    br = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='br')
    Wz = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wz')
    Uz = T.shared(orthogonal((gru_dim, gru_dim)), name='Uz')
    bz = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='bz')
    Wg = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wg')
    Ug = T.shared(orthogonal((gru_dim, gru_dim)), name='Ug')
    bg = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='bg')
    W_fc2 = T.shared(glorot_uniform((gru_dim, 4)) if not zero_tail_fc else NP.zeros((gru_dim, 4), dtype=T.config.floatX), name='W_fc2')
    b_fc2 = T.shared(NP.zeros((4,), dtype=T.config.floatX), name='b_fc2')
    ### NETWORK PARAMETERS END

    return conv_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2

def build_parser():
    parser = ap.ArgumentParser(description='Trains a RNN tracker')
    parser.add_argument('--dataDir', help='Directory of trajectory model', type=str, default='/home/jccaicedo/localization-agent/notebooks')
    parser.add_argument('--batch_size', help='Number of elements in batch', type=int, default=32)
    parser.add_argument('--conv_nr_filters', help='Number of feature filters', type=int, default=32)
    parser.add_argument('--conv_filter_row', help='Rows of feature filters', type=int, default=10)
    parser.add_argument('--conv_filter_col', help='Columns of feature filters', type=int, default=10)
    parser.add_argument('--conv_stride', help='Convolutional stride', type=int, default=5)
    parser.add_argument('--img_row', help='Image rows', type=int, default=100)
    parser.add_argument('--img_col', help='Image cols', type=int, default=100)
    parser.add_argument('--gru_dim', help='Dimension of GRU state', type=int, default=256)
    parser.add_argument('--seq_len', help='Lenght of sequences', type=int, default=60)
    parser.add_argument('--model_name', help='Name of model file', type=str, default='model.pkl')
    parser.add_argument('--zero_tail_fc', help='', type=bool, default=False)
    parser.add_argument('--test', help='', type=bool, default=False)
    parser.add_argument('--use_cudnn', help='Use CUDA CONV or THEANO', type=bool, default=False)
    parser.add_argument('--learning_rate', help='SGD learning rate', type=float, default=0.0005)
    parser.add_argument('--epochs', help='Number of epochs with 32000 example sequences each', type=int, default=1)
    return parser

### Utility functions end

if __name__ == '__main__':
    ### CONFIGURATION BEGIN
    parser = build_parser()
    args = parser.parse_args()
    globals().update(vars(args))
    ### CONFIGURATION END

    ### Computed hyperparameters begin
    conv_output_dim = ((img_row - conv_filter_row) / conv_stride + 1) * \
        ((img_col - conv_filter_col) / conv_stride + 1) * \
        conv_nr_filters
    gru_input_dim = conv_output_dim + 4
    ### Computed hyperparameters end

    train, tester, params = build(batch_size, gru_input_dim, gru_dim, conv_output_dim, conv_nr_filters, conv_filter_row, conv_filter_col, conv_stride, zero_tail_fc, test, use_cudnn, learning_rate)

    try:
        f = open(model_name, "rb")
        param_saved = cPickle.load(f)
        for _p, p in zip(params, param_saved):
            _p.set_value(p)
    except IOError:
        pass

    print 'Generating dataset'
    generator = GG.GaussianGenerator(dataDir=dataDir, seqLength=seq_len, imageSize=img_row, grayscale=True)
    print 'START'


    try:
        N = 32000/batch_size # Constant number of example sequences per epoch
        for i in range(0, epochs):
            train_cost = test_cost = 0
            for j in range(0, N):
                st = time.time()
                data, label = generator.getBatchInParallel(batch_size)
                clock('Simulations',st)

                st = time.time()
                print 'Initial data shape:', data.shape
                if generator.grayscale:
                    data = data[:, :, NP.newaxis, :, :]
                else:
                    data = data.transpose((0,1,4,2,3))
                data /= 255.0
                print 'Final data shape:', data.shape
                label = label / (img_row / 2.) - 1.
                clock('Normalization',st)

                # We can also implement a 'replay memory' here to store previous simulations and reuse them again later during training.
                # The basic idea is to store sequences in a tensor of a predefined capacity, and when it's full we start sampling sequences
                # from the memory with certain probability. The rest of the time new sequences are simulated. This could save some processing time.

                st = time.time()
                cost, bbox_seq = train(seq_len, data, label[:, 0, :], label)
                clock('Training',st)
                '''left = NP.max([bbox_seq[:, :, 0], label[:, :, 0]], axis=0)
                top = NP.max([bbox_seq[:, :, 1], label[:, :, 1]], axis=0)
                right = NP.min([bbox_seq[:, :, 2], label[:, :, 2]], axis=0)
                bottom = NP.min([bbox_seq[:, :, 3], label[:, :, 3]], axis=0)
                intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
                label_area = (label[:, :, 2] - label[:, :, 0]) * (label[:, :, 2] - label[:, :, 0] > 0) * (label[:, :, 3] - label[:, :, 1]) * (label[:, :, 3] - label[:, :, 1] > 0)
                predict_area = (bbox_seq[:, :, 2] - bbox_seq[:, :, 0]) * (bbox_seq[:, :, 2] - bbox_seq[:, :, 0] > 0) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1]) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1] > 0)
                union = label_area + predict_area - intersect'''
                print i, j, cost
                train_cost += cost
                '''data, label = generator.getBatch(batch_size)
                data = data[:, :, NP.newaxis, :, :] / 255.0
                label = label / (img_row / 2.) - 1.
                cost, bbox_seq = tester(seq_len, data, label[:, 0, :], label)
                left = NP.max([bbox_seq[:, :, 0], label[:, :, 0]], axis=0)
                top = NP.max([bbox_seq[:, :, 1], label[:, :, 1]], axis=0)
                right = NP.min([bbox_seq[:, :, 2], label[:, :, 2]], axis=0)
                bottom = NP.min([bbox_seq[:, :, 3], label[:, :, 3]], axis=0)
                intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
                label_area = (label[:, :, 2] - label[:, :, 0]) * (label[:, :, 2] - label[:, :, 0] > 0) * (label[:, :, 3] - label[:, :, 1]) * (label[:, :, 3] - label[:, :, 1] > 0)
                predict_area = (bbox_seq[:, :, 2] - bbox_seq[:, :, 0]) * (bbox_seq[:, :, 2] - bbox_seq[:, :, 0] > 0) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1]) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1] > 0)
                union = label_area + predict_area - intersect
                print i, j, cost
                test_cost += cost
                iou = intersect / union
                print NP.average(iou, axis=0)       # per frame
                print NP.average(iou, axis=1)       # per batch'''
            print 'Epoch average loss (train, test)', train_cost / 2000, test_cost / 2000
            dump_params(model_name + str(i), params)
    except KeyboardInterrupt:
        if not test:
            print 'Saving...'
            dump_params(model_name, params)
