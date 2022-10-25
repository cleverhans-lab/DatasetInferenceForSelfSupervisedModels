CUDA_VISIBLE_DEVICES=0
for NUMQUERIES in 500 1000 2000 3000 4000 5000 7000 8000 9000 10000 20000 30000 40000 50000 ; do
  python dist.py  --samplesize 1000 --dataset 'svhn' --datasetsteal 'cifar10' --archstolen 'resnet34' --num_queries $NUMQUERIES --datasetrand 'cifar100' --epochsrand 100 --archrand 'resnet18'
done
