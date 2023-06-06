#python -m torch.distributed.launch --nproc_per_node=2 train.py --id SAC_AD-clear --batch_size 72
CUDA_VISIBLE_DEVICES=1,2 python train.py --id SAC-AD-clear --batch_size 256
#CUDA_VISIBLE_DEVICES=0
