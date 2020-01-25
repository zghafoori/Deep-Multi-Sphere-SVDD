# Deep-Multi-Sphere-SVDD

This repository provides the implementation of "Deep Multi-sphere Support Vector Data Description” SIAM International Conference on Data Mining (SDM20). If you use our work, please also cite our paper.

## Abstract
Deep learning is increasingly used for unsupervised feature extraction and anomaly detection in big datasets. Most deep learning based anomaly detection techniques separately train a neural network for feature extraction, then apply a traditional anomaly detection method on the extracted features. These hybrid techniques have achieved higher accuracy than traditional anomaly detection methods and reconstruction-error-based deep autoencoders. However, recent research demonstrates that jointly optimising the objectives of the deep network and the anomaly detection technique in a hybrid architecture substantially improves detection performance. Existing methods that use this objective assume that the normal (i.e., non-anomalous) data comes from a single distribution. In this paper, we show that violation of this assumption negatively affects performance of these methods and creates model bias in the favour of anomalies. We propose Deep Multi-sphere Support Vector Data Description, which jointly optimises the objectives of the deep network and anomaly detection. It generates useful and discriminative features by embeding normal data with a multi-modal distribution into multiple data-enclosing hyper- spheres with minimum volume. We empirically show that our proposed method outperforms state-of-the-art shallow and deep anomaly detection methods.

## Code example for running Deep Multi-Sphere SVDD for three datasets
######sh scripts/mnist_msvdd.sh gpu mnist_01vsall_msvdd_d32_c100 0 adam 0.0001 100 1 '[0,1]' 'range(2,10)' 100 0.1 32 mnist-msvdd;
sh scripts/cifar10_msvdd.sh gpu cifar10_19vsall_msvdd_c100 0 adam 0.0001 100 '[1,9]' 'range(2,8)' 100 0.1 cifar10-msvdd;
sh scripts/mobiFall_msvdd.sh gpu mobi_msvdd_d10_c100 0 adam 0.0001 100 1 'np.append([2,5,6,7],range(9,14))' '[1,3,4,8]' 100 0.1 10 mobi-msvdd;

## Disclosure
This implementation is based on the repository https://github.com/lukasruff/Deep-SVDD, which implements Deep SVDD. Instructions for installation are similar to what is explained in this repository. 
