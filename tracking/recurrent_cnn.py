import argparse as ap

import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG

import caffe

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

### Recurrent step
# img: of shape (batch_size, nr_channels, img_rows, img_cols)
def _step(act1, prev_bbox, state, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2, batch_size, conv_output_dim):
	# of (batch_size, nr_filters, some_rows, some_cols)
	flat1 = TT.reshape(act1, (batch_size, conv_output_dim))
	gru_in = TT.concatenate([flat1, prev_bbox], axis=1)
	gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
	gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
	gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
	gru_h = (1-gru_z) * state + gru_z * gru_h_
	bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)
	return bbox, gru_h

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

def setup(batch_size, seq_len, img_row, img_col, deployPath='/home/jccaicedo/data/simulations/cnns/googlenet/deploy.prototxt', modelPath='/home/jccaicedo/data/simulations/cnns/googlenet/bvlc_googlenet.caffemodel', caffe_root='/home/jccaicedo/caffe/'):
    print 'Creating Net object'
    caffe.set_mode_gpu()
    net = caffe.Net(deployPath, modelPath, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    print 'Creating Transformer'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', NP.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 50
    # TODO: correct input shape
    print 'Reshaping input from {} to {}'.format(net.blobs['data'].data.shape, (batch_size*seq_len,3,img_row,img_col))
    net.blobs['data'].reshape(batch_size*seq_len,3,img_row,img_col)

    return net, transformer

def forward(net, transformer, images, layerKey):
    #TODO: check equivalence with caffe.io.load_image(imagePath)
    print 'Forwarding images with shape {}'.format(images.shape)
    net.blobs['data'].data[...] = NP.array([transformer.preprocess('data', image) for image in images])
    out = net.forward(blobs=[layerKey])
    return out[layerKey]

def dump_params(model_name, params):
    f = open(model_name, "wb")
    cPickle.dump(map(lambda x: x.get_value(), params), f)
    f.close()

def build(batch_size, gru_input_dim, gru_dim, conv_output_dim, zero_tail_fc, test):
    print 'Building network'

    # imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
    imgs = tensor5()
    starts = TT.matrix()

    Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2 = init_params(gru_input_dim, gru_dim, zero_tail_fc)
    params = [Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2]

    # Move the time axis to the top
    sc, _ = T.scan(_step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, TT.zeros((batch_size, gru_dim))], non_sequences=params+[batch_size,conv_output_dim], strict=True)

    bbox_seq = sc[0].dimshuffle(1, 0, 2)

    # targets: of shape (batch_size, seq_len, 4)
    targets = TT.tensor3()
    seq_len_scalar = TT.scalar()

    cost = ((targets - bbox_seq) ** 2).sum() / batch_size / seq_len_scalar

    print 'Building optimizer'

    train = T.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], updates=rmsprop(cost, params) if not test else None, allow_input_downcast=True)
    tester = T.function([seq_len_scalar, imgs, starts, targets], [cost, bbox_seq], allow_input_downcast=True)
    
    return train, tester, params

def init_params(gru_input_dim, gru_dim, zero_tail_fc):
    print 'Initializing parameters'

    ### NETWORK PARAMETERS BEGIN
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

    return Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2

def build_parser():
    parser = ap.ArgumentParser(description='Trains a RNN tracker')
    parser.add_argument('--dataDir', help='Directory of trajectory model', type=str, default='/home/jccaicedo/localization-agent/notebooks')
    parser.add_argument('--batch_size', help='Number of elements in batch', type=int, default=4)
    parser.add_argument('--layer_key', help='Key string of layer name to use as features', type=str, default='inception_5b/output')
    parser.add_argument('--img_row', help='Image rows', type=int, default=224)
    parser.add_argument('--img_col', help='Image cols', type=int, default=224)
    parser.add_argument('--gru_dim', help='Dimension of GRU state', type=int, default=256)
    parser.add_argument('--seq_len', help='Lenght of sequences', type=int, default=60)
    parser.add_argument('--model_name', help='Name of model file', type=str, default='model.pkl')
    parser.add_argument('--zero_tail_fc', help='', type=bool, default=False)
    parser.add_argument('--test', help='', type=bool, default=False)
    return parser

### Utility functions end

if __name__ == '__main__':
    ### CONFIGURATION BEGIN
    parser = build_parser()
    args = parser.parse_args()
    globals().update(vars(args))
    ### CONFIGURATION END

    net, transform = setup(batch_size, seq_len, img_row, img_col)

    ### Computed hyperparameters begin
    conv_nr_filters, conv_filter_row, conv_filter_col = net.blobs[layer_key].data.shape[-3:]
    print 'Shape of layer {}: {}'.format(layer_key, (conv_nr_filters, conv_filter_row, conv_filter_col))
    conv_output_dim = conv_nr_filters*conv_filter_row*conv_filter_col
    gru_input_dim = conv_output_dim + 4
    ### Computed hyperparameters end

    train, tester, params = build(batch_size, gru_input_dim, gru_dim, conv_output_dim, zero_tail_fc, test)

    try:
        f = open(model_name, "rb")
        param_saved = cPickle.load(f)
        for _p, p in zip(params, param_saved):
            _p.set_value(p)
    except IOError:
        pass

    print 'Generating dataset'

    generator = GG.GaussianGenerator(dataDir=dataDir, seqLength=seq_len, imageSize=img_row, grayscale=False)
    print 'START'

    try:
        for i in range(0, 50):
            train_cost = test_cost = 0
            for j in range(0, 2000):
                st = time.time()
                data, label = generator.getBatch(batch_size)
                clock('Simulations',st)

                st = time.time()
                print 'Initial data shape:', data.shape
                if generator.grayscale:
                    data = data[:, :, NP.newaxis, :, :]
                data /= 255.0
                print 'Final data shape:', data.shape
                label = label / (img_row / 2.) - 1.
                clock('Normalization',st)

                            # We can also implement a 'replay memory' here to store previous simulations and reuse them again later during training.
                            # The basic idea is to store sequences in a tensor of a predefined capacity, and when it's full we start sampling sequences
                            # from the memory with certain probability. The rest of the time new sequences are simulated. This could save some processing time.

                st = time.time()
                activations = forward(net, transform, data.reshape(-1, data.shape[-3], data.shape[-2], data.shape[-1]), layer_key)
                activations = activations.reshape(batch_size, seq_len, activations.shape[-3], activations.shape[-2], activations.shape[-1])
                clock('Activations', st)

                st = time.time()
                print 'Sequence length: {} Activations shape: {} Labels shape: {}'.format(seq_len, activations.shape, label.shape)
                cost, bbox_seq = train(seq_len, activations, label[:, 0, :], label)
                clock('Training',st)
                print i, j, cost
                train_cost += cost
            print 'Epoch average loss (train, test)', train_cost / 2000, test_cost / 2000
            dump_params(model_name + str(i), params)
    except KeyboardInterrupt:
        if not test:
            print 'Saving...'
            dump_params(model_name, params)
