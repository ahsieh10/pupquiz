#!/bin/bash
#SBATCH -J TrainModel
 
#SBATCH -n 30
#SBATCH --mem=32G
#SBATCH -t 24:00:00
 
#SBATCH -o train_stats.txt
 
#SBATCH --mail-type=END
#SBATCH --mail-user=pranav_mahableshwarkar@brown.edu
# SBATCH --mail-user=joseph_dodson@brown.edu
 
module load miniconda/4.10
 
source /gpfs/runtime/opt/miniconda/4.10/etc/profile.d/conda.sh
conda activate csci1470
 
python3 train.py
