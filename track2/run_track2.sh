#!/bin/bash
#SBATCH -J track2
#SBATCH -o track2.out
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6
# --pt_name fnlp/bart-base-chinese\

# python ./src/main.py\
torchrun --nproc_per_node 1 --master_port 33334 ./src/main.py \
        --seed 69\
        --train_json data/train.json\
        --test_json data/test.json\
        --data_cache dataset_cache/\
        --pt_cache pt_model/\
        --num_train_epochs 30\
        --cache_dir ./model/ft_ccl\
        --lr 5e-5\
        --do_train\
        --batch_size 32\
	--checkpoint_dir ./model/pt_ccl/checkpoint.tar
