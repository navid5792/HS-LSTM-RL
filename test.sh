#!/bin/bash
#SBATCH --account=def-mercer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH -c 5  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=12G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-70:00
#SBATCH -o outfileNew29
#SBATCH -e errorfileNew29

python -u main.py --Newcriticmodelname model_test61New_29.pickle --Newactormodelname model_actor61New_29.pickle --optimizer sgd,lr=0.29 > outfileNew29.out





