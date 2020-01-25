#!/usr/bin/env bash

xp_dir=log/$1
seed=$2
pca=$3
mobiFall_normal=$4
mobiFall_outlier=$5
info_file=$6;

mkdir $xp_dir;

# mobiFall training
python baseline_kde.py --xp_dir $xp_dir --dataset mobiFall --kernel gaussian --seed $seed --GridSearchCV 1 \
    --out_frac 0 --unit_norm_used l1 --pca $pca --mobiFall_val_frac 0 --mobiFall_normal $mobiFall_normal \
    --mobiFall_outlier $mobiFall_outlier --plot_examples 1 --info_file $info_file;
