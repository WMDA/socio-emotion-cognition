#! /bin/bash

#SBATCH --job-name=tv
#SBATCH --output=/data/project/BEACONB/logs/spacenet_best.out
#SBATCH --export=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G

source /software/system/modules/latest/init/bash
module use /software/system/modules/NaN/generic
module purge
module load nan
module load miniconda/3

echo "Running on $HOSTNAME"
conda activate neuroimaging
python3 /data/project/BEACONB/task_fmri/socio-emotion-cognition/task_fmri/mvpa/build_spacenet_models.py