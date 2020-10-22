#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=64G

module load python/3.7.0
module load scipy-stack

source ~/pytorch/bin/activate

python multi_frame_dip.py --config config.cfg
