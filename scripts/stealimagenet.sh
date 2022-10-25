CUDA_VISIBLE_DEVICES=0,1,2,3
python stealsimsiam.py --world-size -1 --rank 0  --pretrained models/checkpoint_0099-batch256.pth.tar --data /home/nicolas/data/imagenet --batch-size 512 --lr 0.1 --losstype 'infonce' --datasetsteal 'imagenet' --workers 8 --num_queries 250000 --useaug 'False' --temperature 0.25
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 250000 --losstype 'infonce' --dataset 'cifar10' --modeltype 'stolen' --workers 8
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 250000 --losstype 'infonce' --dataset 'cifar100' --modeltype 'stolen' --workers 8
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 250000 --losstype 'infonce' --dataset 'stl10' --modeltype 'stolen' --workers 8
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 250000 --losstype 'infonce' --dataset 'svhn' --modeltype 'stolen' --workers 8
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 250000 --losstype 'infonce' --dataset 'fmnist' --modeltype 'stolen' --workers 8

