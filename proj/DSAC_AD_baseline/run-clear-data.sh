#python -m torch.distributed.launch --nproc_per_node=2 train.py --id SAC_AD-clear --batch_size 72
CUDA_VISIBLE_DEVICES=5 python train.py --id SAC-AD-clear --batch_size 200
#CUDA_VISIBLE_DEVICES=0
