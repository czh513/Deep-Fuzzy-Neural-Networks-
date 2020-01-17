
out_dir=$1

train() {
    name=$1
    second_param_onwards="${@:2}"
    echo "Training $name"
    python -u train.py cifar-10 --vgg_name=VGG11 --use_batchnorm=True --modification_start_layer=0 --n_epochs=80 --out_path=$out_dir/$name.pkl $second_param_onwards &> $out_dir/$name.log
}

relog_conf="--use_relog=True --log_strength_start=-0.8 --log_strength_inc=0.0001 --log_strength_stop=2"
elliptical_conf="--use_elliptical=True --curvature_multiplier_start=-0.4 --curvature_multiplier_inc=0.0001 --curvature_multiplier_stop=0.5"
maxout_conf="--use_maxout=max --max_folding_factor=2"
minmaxout_conf="--use_maxout=minmax --min_folding_factor=2 --max_folding_factor=2"
maxfit_conf="--regularization=max-fit --l1=0.1 --bias_l1=0.5 --regularization_start_epoch=52"
mse_conf="--use_mse=True --mse_weighted=False"
bce_conf="--use_bce=True"
