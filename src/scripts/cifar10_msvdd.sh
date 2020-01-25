#!/usr/bin/env bash

device=$1
xp_dir=log/$2
seed=$3
solver=$4
lr=$5
n_epochs=$6
cifar10_normal=$7
cifar10_outlier=$8
n_cluster=$9
nu=${10}
info_file=${11}


mkdir $xp_dir;

# CIFAR-10 training
python baseline.py --dataset cifar10 --solver $solver --loss msvdd --lr $lr --lr_drop 1 --lr_drop_in_epoch 50 \
    --seed $seed --lr_drop_factor 10 --block_coordinate 0 --center_fixed 1 \
    --use_batch_norm 1 --pretrain 1 --batch_size 200 --n_epochs $n_epochs --device $device \
    --xp_dir $xp_dir --leaky_relu 1 --weight_decay 1 --C 1e6 --reconstruction_penalty 0 --c_mean_init 1 \
    --hard_margin 1 --nu $nu --out_frac 0 --weight_dict_init 0 --unit_norm_used l1 --gcn 1 --cifar10_bias 0 \
    --cifar10_rep_dim 128 --cifar10_normal $cifar10_normal --cifar10_outlier $cifar10_outlier --nnet_diagnostics 0 \
    --e1_diagnostics 0 --n_cluster $n_cluster --plot_examples 1 --info_file $info_file;
