from datasets.base import DataLoader
from datasets.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, learn_dictionary_new, pca
from datasets.modules import addConvModule, addDeConvModule
from utils.visualization.mosaic_plot import plot_mosaic
from utils.misc import flush_last_line
from config import Configuration as Cfg

import numpy as np


class mobiFall_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "mobiFall"

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

        X,y = load_mobiFall_data('%smobiFall/train_mobiFall' %
                              self.data_path)
        
        X_test,y_test = load_mobiFall_data('%smobiFall/test_mobiFall' %
                              self.data_path)

        if Cfg.ad_experiment:
            
            # set normal and anomalous class
            normal = eval(Cfg.mobiFall_normal)
            outliers = eval(Cfg.mobiFall_outlier)

            # extract normal and anomalous class
            X_norm, X_out, y_norm, y_out, idx_norm, idx_out = extract_norm_and_out(X, y, normal=normal, outlier=outliers)

            # reduce outliers to fraction defined
            n_norm = len(y_norm)
            n_out = int(np.ceil(Cfg.out_frac * n_norm / (1 - Cfg.out_frac)))

            # shuffle to obtain random validation splits
            np.random.seed(self.seed)
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))

            # split into training and validation set
            n_norm_split = int(Cfg.mobiFall_val_frac * n_norm)
            n_out_split = int(Cfg.mobiFall_val_frac * n_out)
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

            # rescale to [0,1] (w.r.t. min and max in train data)
            rescale_to_unit_interval(self._X_train, self._X_val, self._X_test)

            # PCA
            if Cfg.pca:
                self._X_train, self._X_val, self._X_test = pca(self._X_train, self._X_val, self._X_test, 0.95)

        flush_last_line()
        print("Data loaded.")

    def build_architecture(self, nnet):

        # implementation of the encoder network architectures
        fs = (5,5)
        fn = [64,32]
        mps = (2,2)
        hn = fn[-1]*5
        # increase number of parameters if dropout is used
        if Cfg.dropout_architecture:
            units_multiplier = 2
        else:
            units_multiplier = 1

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary_new(nnet.data._X_train, n_filters=fn[0], filter_shape=fs, n_sample=500)
            W1_init = np.reshape(W1_init,(fn[0],self._X_train.shape[1])+fs)
            plot_mosaic(W1_init, title="First layer filters initialization",
                        canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None
            
        # build architecture
        nnet.addInputLayer(shape=(None,) + self._X_train.shape[1:])

        addConvModule(nnet,
                      num_filters=fn[0] * units_multiplier,
                      filter_size=fs,
                      W_init=W1_init,
                      bias=Cfg.mobiFall_bias,
                      pool_size=mps,
                      use_batch_norm=Cfg.use_batch_norm,
                      dropout=Cfg.dropout,
                      p_dropout=0.2)

        addConvModule(nnet,
                      num_filters=fn[1] * units_multiplier,
                      filter_size=fs,
                      bias=Cfg.mobiFall_bias,
                      pool_size=mps,
                      use_batch_norm=Cfg.use_batch_norm,
                      dropout=Cfg.dropout)

        if Cfg.dropout:
            nnet.addDropoutLayer()

        if Cfg.mobiFall_bias:
            nnet.addDenseLayer(num_units=hn * units_multiplier)
        else:
            nnet.addDenseLayer(num_units=hn * units_multiplier,
                               b=None)

        if Cfg.dropout:
            nnet.addDropoutLayer()

        if Cfg.mobiFall_bias:
            nnet.addDenseLayer(num_units=Cfg.mobiFall_rep_dim * units_multiplier)
        else:
            nnet.addDenseLayer(num_units=Cfg.mobiFall_rep_dim * units_multiplier,
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
        fs = (5,5)
        fn = [64,32]
        mps = (2,2)
        hn = fn[-1]*5

        ##############################
        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary_new(nnet.data._X_train, n_filters=fn[0], filter_shape=fs, n_sample=500)
            W1_init = np.reshape(W1_init,(fn[0],self._X_train.shape[1])+fs)
            plot_mosaic(W1_init, title="First layer filters initialization",
                        canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        nnet.addInputLayer(shape=(None,) + self._X_train.shape[1:])
        
        addConvModule(nnet,
                      num_filters=fn[0],
                      filter_size=fs,
                      W_init=W1_init,
                      bias=Cfg.mobiFall_bias,
                      pool_size=mps,
                      use_batch_norm=Cfg.use_batch_norm)

        addConvModule(nnet,
                      num_filters=fn[1],
                      filter_size=fs,
                      bias=Cfg.mobiFall_bias,
                      pool_size=mps,
                      use_batch_norm=Cfg.use_batch_norm)
        
        h_units = np.prod(nnet.all_layers[-1].output_shape[1:])
        d_r = nnet.all_layers[-1].output_shape[1:]

        if Cfg.mobiFall_bias:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn)
        else:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn, b=None)

        # Code Layer
        if Cfg.mobiFall_bias:
            nnet.addDenseLayer(num_units=Cfg.mobiFall_rep_dim)
        else:
            nnet.addDenseLayer(num_units=Cfg.mobiFall_rep_dim, b=None)
        nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer

        if Cfg.mobiFall_bias:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn)
        else:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=hn, b=None)
            
        if Cfg.mobiFall_bias:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=h_units)
        else:
            nnet.addDenseLayer(use_batch_norm=Cfg.use_batch_norm, num_units=h_units, b=None)

        nnet.addReshapeLayer(shape=([0], h_units/np.prod(d_r[1:]), d_r[1], d_r[2]))

        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        addConvModule(nnet,
                      num_filters=fn[1],
                      filter_size=fs,
                      bias=Cfg.mobiFall_bias,
                      pool_size=mps,
                      use_batch_norm=Cfg.use_batch_norm,
                      upscale=True)
        
        addConvModule(nnet,
                      num_filters=fn[0],
                      filter_size=fs,
                      bias=Cfg.mobiFall_bias,
                      pool_size=mps,
                      use_batch_norm=Cfg.use_batch_norm,
                      upscale=True)

        # reconstruction
        if Cfg.mobiFall_bias:
            nnet.addConvLayer(num_filters=1,
                              filter_size=fs,
                              pad='same')
        else:
            nnet.addConvLayer(num_filters=3,
                              filter_size=fs,
                              pad='same',
                              b=None)

def load_mobiFall_data(filename):

    X = np.loadtxt(filename+".txt",delimiter=',').astype(np.float32)
    Y = np.loadtxt(filename+"_labels.txt",delimiter=',').astype(np.int32)
    X = X.reshape(-1, 3, 20, 20).astype(np.float32)

    return X,Y
