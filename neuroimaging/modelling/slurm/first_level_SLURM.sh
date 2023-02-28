#! /bin/bash

#SBATCH --job-name=first_level_modelling
#SBATCH --output=/data/project/BEACONB/logs/%j.out
#SBATCH --export=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --array=93

source /software/system/modules/latest/init/bash
module use /software/system/modules/NaN/generic
module purge
module load nan

module load spm/12-7771
module load miniconda/3

INDEX=/data/project/BEACONB/task_fmri/socio-emotion-cognition/.participants_t1
PARTICIPANT="`awk FNR==$SLURM_ARRAY_TASK_ID $INDEX`"

echo "Running on $HOSTNAME"
echo $PARTICIPANT
conda activate neuroimaging
python3 /data/project/BEACONB/task_fmri/socio-emotion-cognition/neuroimaging/first_level/first_level.py $PARTICIPANT
echo "Complete"