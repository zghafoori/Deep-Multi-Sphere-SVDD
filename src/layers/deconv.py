import lasagne.layers
#import theano.tensor as T
#from lasagne import nonlinearities.LeakyRectify


class deConvLayer(lasagne.layers.TransposedConv2DLayer):

    # for convenience
    isdense, isbatchnorm, isdropout, ismaxpool, isactivation = (False,) * 5
    isconv = True

    def __init__(self, incoming_layer, num_filters, filter_size, stride=(1, 1),
                 crop=0, W=lasagne.init.GlorotUniform(gain='relu'),
                 b=lasagne.init.Constant(0.), flip_filters=False, name=None):

        lasagne.layers.TransposedConv2DLayer.__init__(self, incoming_layer, num_filters,
                                            filter_size, name=name,
                                            stride=stride, crop=crop,
                                            untie_biases=False, W=W, b=b,
                                            nonlinearity=None,
                                            flip_filters=flip_filters)#,
                                            #convolution=T.nnet.conv2d)

        self.inp_ndim = 4
        self.use_dc = False
        