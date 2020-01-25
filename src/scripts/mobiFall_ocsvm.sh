#!/usr/bin/env bash

xp_dir=log/$1
seed=$2
nu=$3
pca=$4
mobiFall_normal=$5
mobiFall_outlier=$6
info_file=$7

mkdir $xp_dir;

# mobiFall training
python baseline_ocsvm.py --xp_dir $xp_dir --dataset mobiFall --loss OneClassSVM --seed $seed --kernel rbf --nu $nu \
    --GridSearchCV 1 --out_frac 0 --unit_norm_used l1 --pca $pca --mobiFall_val_frac 0 --mobiFall_normal $mobiFall_normal \
    --mobiFall_outlier $mobiFall_outlier --plot_examples 1 --info_file $info_file;
