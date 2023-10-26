#!/bin/bash
#SBATCH -J track3
#SBATCH -o bart_base.out
#SBATCH -n 1
#SBATCH --gres=gpu:a100-sxm-80gb:4
#SBATCH -t 2-00:00:00

export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=5,6,7

# python ./src/main.py\
torchrun --nproc_per_node 1 --master_port 33343 ./src/main.py \
        --seed 47\
        --train_json data/train.json\
        --test_json data/val.json\
        --data_cache dataset_cache/\
        --pt_name ./pt_model/bart\
        --pt_cache pt_model/\
        --num_train_epochs 40\
        --cache_dir ./model\
        --lr 5e-5\
        --do_train\
        --batch_size 32\
	--checkpoint_dir ./model/day0609_track3/checkpoint.tar
