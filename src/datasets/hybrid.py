from datasets.base import DataLoader
from datasets.preprocessing import extract_norm_and_out
from config import Configuration as Cfg
import numpy as np

class Hybrid_DataLoader(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.seed = Cfg.seed
        self.n_classes = 2

        self.on_memory = True
        Cfg.store_on_gpu = False

        # load data from disk
        self.load_dataset_path()

    def load_dataset_path(self):

	path = '../log/hybrid/'+Cfg.dataset+'/seed_'+str(Cfg.seed)
        print("Loading data from path: ...", path)
	
	self._X_train = np.loadtxt(path+"/repsTrain_ae.txt",delimiter=',')
	self._y_train = np.zeros(len(self._X_train), dtype=np.uint8)
	self._X_val = np.loadtxt(path+"/repsVal_ae.txt",delimiter=',')
	self._y_val = np.zeros(len(self._X_val), dtype=np.uint8)	

	self.n_train = len(self._y_train)
	self.n_val = len(self._y_val)

	# Adjust number of batches
	Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

	# test set
	normal = eval(Cfg.mnist_normal)
	outliers = eval(Cfg.mnist_outlier)
	X_test = np.loadtxt(path+"/repsTest_ae.txt",delimiter=',')
	y_test = np.loadtxt(path+"/ltest_ae.txt",delimiter=',')
	X_norm, X_out, y_norm, y_out, idx_norm, idx_out = extract_norm_and_out(X_test, y_test, normal=normal, outlier=outliers)
            
	#zahra
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
