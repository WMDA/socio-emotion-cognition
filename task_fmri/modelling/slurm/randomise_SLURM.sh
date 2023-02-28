#! /bin/bash

#SBATCH --job-name=randomise
#SBATCH --output=/data/project/BEACONB/logs/%j.out
#SBATCH --export=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

source /software/system/modules/latest/init/bash
module use /software/system/modules/NaN/generic
module purge
module load nan

module load fsl/6.0.5.2
module load miniconda/3


echo "Running on $HOSTNAME"
conda activate neuroimaging
python3 /data/project/BEACONB/task_fmri/socio-emotion-cognition/neuroimaging/modelling/second_level.py