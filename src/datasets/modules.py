from config import Configuration as Cfg
from lasagne.init import GlorotUniform, Constant


def addConvModule(nnet, num_filters, filter_size, pad='same', W_init=None, bias=True, pool_size=(2,2),
                  use_batch_norm=False, dropout=False, p_dropout=0.5, upscale=False):
    """
    add a convolutional module (convolutional layer + (leaky) ReLU + MaxPool) to the network  
    """

    if W_init is None:
        W = GlorotUniform(gain=(2/(1+0.01**2)) ** 0.5)  # gain adjusted for leaky ReLU with alpha=0.01
    else:
        W = W_init

    if bias is True:
        b = Constant(0.)
    else:
        b = None

    # build module
    if dropout:
        nnet.addDropoutLayer(p=p_dropout)

    nnet.addConvLayer(use_batch_norm=use_batch_norm,
                      num_filters=num_filters,
                      filter_size=filter_size,
                      pad=pad,
                      W=W,
                      b=b)

    if Cfg.leaky_relu:
        nnet.addLeakyReLU()
    else:
        nnet.addReLU()

    if upscale:
        nnet.addUpscale(scale_factor=pool_size)
    else:
        nnet.addMaxPool(pool_size=pool_size)


def addDeConvModule(nnet, num_filters, filter_size, crop=0, W_init=None, bias=True, pool_size=(2,2),
                  use_batch_norm=False, dropout=False, p_dropout=0.5, upscale=False, flip_filters=False, activationFlag=True):
    """
    add a convolutional module (convolutional layer + (leaky) ReLU + MaxPool) to the network
    """

    if W_init is None:
        W = GlorotUniform(gain=(2/(1+0.01**2)) ** 0.5)  # gain adjusted for leaky ReLU with alpha=0.01
    else:
        W = W_init

    if bias is True:
        b = Constant(0.)
    else:
        b = None

    # build module
    if upscale:
        nnet.addUpscale(scale_factor=pool_size)

    if dropout:
        nnet.addDropoutLayer(p=p_dropout)

    nnet.addDeConvLayer(activationFlag=activationFlag, use_batch_norm=use_batch_norm,
                      num_filters=num_filters,
                      filter_size=filter_size,
                      crop=crop,
                      W=W,
                      b=b,
                      flip_filters=flip_filters)

    if Cfg.leaky_relu:
        nnet.addLeakyReLU()
    else:
        nnet.addReLU()
