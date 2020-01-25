from datasets.base import DataLoader
from datasets.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from datasets.modules import addConvModule
from utils.visualization.mosaic_plot import plot_mosaic
from utils.misc import flush_last_line
from config import Configuration as Cfg

import gzip
import numpy as np


class MNIST_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "mnist"

        self.n_train = 50000
        self.n_val = 10000
        self.n_test = 10000

        self.seed = Cfg.seed

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "../data/"

        self.on_memory = True
        Cfg.store_on_gpu = False

        # load data from disk
        self.load_data()

    def check_specific(self):

        # store primal variables on RAM
        assert not(Cfg.store_on_gpu)

    def load_data(self, original_scale=False):

        print("Loading data...")

        X = load_mnist_images('%strain-images-idx3-ubyte.gz' %
                              self.data_path)
        y = load_mnist_labels('%strain-labels-idx1-ubyte.gz' %
                              self.data_path)
        X_test = load_mnist_images('%st10k-images-idx3-ubyte.gz' %
                                   self.data_path)
        y_test = load_mnist_labels('%st10k-labels-idx1-ubyte.gz' %
                                   self.data_path)

        if Cfg.ad_experiment:

            # set normal and anomalous class
            normal = eval(Cfg.mnist_normal)
            outliers = eval(Cfg.mnist_outlier)

            # extract normal and anomalous class
            X_norm, X_out, y_norm, y_out, _, _ = extract_norm_and_out(X, y, normal=normal, outlier=outliers)

            # reduce outliers to fraction defined
            n_norm = len(y_norm)
            n_out = int(np.ceil(Cfg.out_frac * n_norm / (1 - Cfg.out_frac)))

            # shuffle to obtain random validation splits
            np.random.seed(self.seed)
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))

            # split into training and validation set
            n_norm_split = int(Cfg.mnist_val_frac * n_norm)
            n_out_split = int(Cfg.mnist_val_frac * n_out)
            self._X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]],
                                            X_out[perm_out[:n_out][n_out_split:]]))
            self._y_train = np.append(y_norm[perm_norm[n_norm_split:]],
                                      y_out[perm_out[:n_out][n_out_split:]])
            self._X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]],
                                          X_out[perm_out[:n_out][:n_out_split]]))
            self._y_val = np.append(y_norm[perm_norm[:n_norm_split]],
                                    y_out[perm_out[:n_out][:n_out_split]])

            # shuffle data (since batches are extracted block-wise)
            self.n_train = len(self._y_train)
            self.n_val = len(self._y_val)
            perm_train = np.random.permutation(self.n_train)
            perm_val = np.random.permutation(self.n_val)
            self._X_train = self._X_train[perm_train]
            self._y_train = self._y_train[perm_train]
            self._X_val = self._X_val[perm_val]
            self._y_val = self._y_val[perm_val]

            # Subset train set such that we only get batches of the same size
            self.n_train = (self.n_train / Cfg.batch_size) * Cfg.batch_size
            subset = np.random.choice(len(self._X_train), self.n_train, replace=False)
            self._X_train = self._X_train[subset]
            self._y_train = self._y_train[subset]

            # Adjust number of batches
            Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

            # test set
            X_norm, X_out, y_norm, y_out, idx_norm, idx_out = extract_norm_and_out(X_test, y_test, normal=normal, outlier=outliers)
            
            yo_norm = y_test[idx_norm]
            yo_out = y_test[idx_out]
            self._yo_test = np.append(yo_norm, yo_out)

            self._X_test = np.concatenate((X_norm, X_out))
            self._y_test = np.append(y_norm, y_out)
            perm_test = np.random.permutation(len(self._y_test))
            self._X_test = self._X_test[perm_test]
            self._y_test = self._y_test[perm_test]
            self._yo_test = self._yo_test[perm_test]
            self.n_test = len(self._y_test)

        else:
            # split into training, validation, and test sets
            np.random.seed(self.seed)
            perm = np.random.permutation(len(X))

            self._X_train = X[perm[self.n_val:]]
            self._y_train = y[perm[self.n_val:]]
            self._X_val = X[perm[:self.n_val]]
            self._y_val = y[perm[:self.n_val]]
            self._X_test = X_test
            self._y_test = y_test

        # normalize data (if original scale should not be preserved)
        if not original_scale:

            # simple rescaling to [0,1]
            normalize_data(self._X_train, self._X_val, self._X_test, scale=np.float32(255))

            # global contrast normalization
            if Cfg.gcn:
                global_contrast_normalization(self._X_train, self._X_val, self._X_test, scale=Cfg.unit_norm_used)

            # ZCA whitening
            if Cfg.zca_whitening:
                self._X_train, self._X_val, self._X_test = zca_whitening(self._X_train, self._X_val, self._X_test)

            # rescale to [0,1] (w.r.t. min and max in train data)
            rescale_to_unit_interval(self._X_train, self._X_val, self._X_test)

            # PCA
            if Cfg.pca:
                self._X_train, self._X_val, self._X_test = pca(self._X_train, self._X_val, self._X_test, 0.95)
                

        flush_last_line()
        print("Data loaded.")

    def build_architecture(self, nnet):
        
        if Cfg.dropout_architecture:
            units_multiplier = 2
        else:
            units_multiplier = 1

        # implementation of the encoder architectures
        nf = [256,256,128]
        hn = 320 #32

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train, n_filters=nf[0], filter_size=5, n_sample=500)
            plot_mosaic(W1_init, title="First layer filters initialization",
                        canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        # build architecture
        nnet.addInputLayer(shape=(None, 1, 28, 28))

        addConvModule(nnet,
                      num_filters=nf[0] * units_multiplier,
                      filter_size=(5,5),
                      W_init=W1_init,
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm,
                      dropout=Cfg.dropout,
                      p_dropout=0.2)

        addConvModule(nnet,
                      num_filters=nf[1] * units_multiplier,
                      filter_size=(5,5),
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm,
                      dropout=Cfg.dropout)

        addConvModule(nnet,
                      num_filters=nf[2] * units_multiplier,
                      filter_size=(5,5),
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm,
                      dropout=Cfg.dropout)

        if Cfg.dropout:
            nnet.addDropoutLayer()

        if Cfg.mnist_bias:
            nnet.addDenseLayer(num_units=hn * units_multiplier)
        else:
            nnet.addDenseLayer(num_units=hn * units_multiplier,
                               b=None)

        if Cfg.dropout:
            nnet.addDropoutLayer()

        if Cfg.mnist_bias:
            nnet.addDenseLayer(num_units=Cfg.mnist_rep_dim * units_multiplier)
        else:
            nnet.addDenseLayer(num_units=Cfg.mnist_rep_dim * units_multiplier,
                               b=None)
            
        if Cfg.softmax_loss:
            nnet.addDenseLayer(num_units=1)
            nnet.addSigmoidLayer()
        elif Cfg.svdd_loss or Cfg.msvdd_loss:
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
        else:
            raise ValueError("No valid choice of loss for dataset " + self.dataset_name)


    def build_autoencoder(self, nnet):

        # implementation of the autoencoder architectures            
        nf = [256,256,128] #[32,32,16]
        hn = 320 #32

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train, n_filters=nf[0], filter_size=5, n_sample=500)
            plot_mosaic(W1_init, title="First layer filters initialization",
                        canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        # build architecture
        nnet.addInputLayer(shape=(None, 1, 28, 28))

        addConvModule(nnet,
                      num_filters=nf[0],
                      filter_size=(5,5),
                      W_init=W1_init,
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm)

        addConvModule(nnet,
                      num_filters=nf[1],
                      filter_size=(5,5),
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm)

        addConvModule(nnet,
                      num_filters=nf[2],
                      filter_size=(5,5),
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm)

        if Cfg.mnist_bias:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn)
        else:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn, b=None)

        # Code Layer
        if Cfg.mnist_bias:
            nnet.addDenseLayer(num_units=Cfg.mnist_rep_dim)
        else:
            nnet.addDenseLayer(num_units=Cfg.mnist_rep_dim, b=None)
        nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer

        if Cfg.mnist_bias:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn)
        else:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn, b=None)
        nnet.addReshapeLayer(shape=([0], hn/16, 4, 4))
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        addConvModule(nnet,
                      num_filters=nf[2],
                      filter_size=(5,5),
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm,
                      upscale=True)

        addConvModule(nnet,
                      num_filters=nf[1],
                      filter_size=(5,5),
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm,
                      upscale=True)

        # to have the same output dimensions, pad must be 1 here
        addConvModule(nnet,
                      num_filters=nf[0],
                      filter_size=(5,5),
                      pad=1,
                      bias=Cfg.mnist_bias,
                      pool_size=(2,2),
                      use_batch_norm=Cfg.use_batch_norm,
                      upscale=True)

        # reconstruction
        if Cfg.mnist_bias:
            nnet.addConvLayer(num_filters=1,
                              filter_size=(5,5),
                              pad='same')
        else:
            nnet.addConvLayer(num_filters=1,
                              filter_size=(5,5),
                              pad='same',
                              b=None)
        nnet.addSigmoidLayer()
        

def load_mnist_images(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # reshaping and normalizing
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)

    return data


def load_mnist_labels(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data
