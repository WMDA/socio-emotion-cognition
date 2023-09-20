#! /bin/bash
#SBATCH --job-name=first_level_modelling
#SBATCH --output=/data/project/BEACONB/logs/cleaning_images.log
#SBATCH --export=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
source /software/system/modules/latest/init/bash
module use /software/system/modules/NaN/generic
module purge
module load nan
# Load in script dependent modules here
module load miniconda/3
conda activate neuroimaging

python /data/project/BEACONB/task_fmri/socio-emotion-cognition/resting/preprocessing_pipelines/cleaning_images.py