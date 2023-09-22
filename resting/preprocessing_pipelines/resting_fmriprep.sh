#! /bin/bash

#SBATCH --job-name=first_level_modelling
#SBATCH --output=/data/project/BEACONB/logs/%j_resting.out
#SBATCH --export=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=1-11%4
  
source /software/system/modules/latest/init/bash
module use /software/system/modules/NaN/generic
module purge
module load nan

# Load in script dependent modules here
module load fmriprep/22.0.2
module load miniconda/3

INDEX=/data/project/BEACONB/task_fmri/socio-emotion-cognition/.repeat_fmri
PARTICIPANT="`awk FNR==$SLURM_ARRAY_TASK_ID $INDEX`"

# set the working variables
working_dir=/data/project/BEACONB
bids_dir=${working_dir}/CNSCNSD/bids_t2/
outputdata=${working_dir}/resting/preprocessed/
work=${outputdata}work
b_number=$(echo $PARTICIPANT| cut -d "-" -f2)
directory_to_delete=`echo single_subject_${b_number}_wf`

echo "Running on $HOSTNAME"
echo $PARTICIPANT
echo "fmriprep ${bids_dir} ${outputdata} participant -w ${work} --participant_label ${PARTICIPANT} -t rest --output-spaces MNI152NLin2009cAsym:res-2 --write-graph --nthreads 4 --omp-nthreads 4 --mem_mb 20480 --stop-on-first-crash --fs-license-file ${working_dir}/.license.txt"
fmriprep ${bids_dir} ${outputdata} participant --participant_label ${PARTICIPANT} -w ${work} -t rest --output-spaces MNI152NLin2009cAsym:res-2 --write-graph --nthreads 4 --omp-nthreads 4 --mem_mb 20480  --stop-on-first-crash --fs-license-file ${working_dir}/.license.txt

rm -rf ${work}/fmriprep_22_0_wf/${directory_to_delete}