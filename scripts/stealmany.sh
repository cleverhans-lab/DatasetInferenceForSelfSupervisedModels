CUDA_VISIBLE_DEVICES=0,1,2,3
for NUMQUERIES in 500 1000 5000 10000 20000 30000 40000 50000 ; do
  python steal.py --losstype 'infonce' --dataset 'cifar10' --datasetsteal 'svhn' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'infonce' --dataset-test 'cifar10' --datasetsteal 'svhn' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'infonce' --dataset-test 'svhn' --datasetsteal 'svhn' --num_queries $NUMQUERIES
  python linear_eval.py --losstype 'infonce' --dataset-test 'stl10' --datasetsteal 'svhn' --num_queries $NUMQUERIES
done
