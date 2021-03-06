#!/bin/bash
#SBATCH --account=def-mercer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH -c 5  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-15:00
#SBATCH -o outfile40
#SBATCH -e errorfile40

python -u main2.py --Newcriticmodelname model_test61_40.pickle --optimizer sgd,lr=0.40 > outfile40.out
