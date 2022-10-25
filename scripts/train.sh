CUDA_VISIBLE_DEVICES=0,1,2,3
python run.py --dataset 'cifar10'
python linear_eval.py --dataset 'cifar10' --dataset-test 'cifar10' --modeltype 'victim'
python linear_eval.py --dataset 'cifar10' --dataset-test 'svhn' --modeltype 'victim'
python linear_eval.py --dataset 'cifar10' --dataset-test 'stl10' --modeltype 'victim'
