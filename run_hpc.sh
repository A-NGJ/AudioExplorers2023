#!/bin/bash

# ====================
# Use this script to run training on the HPC
# ====================


### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J mel_classify
### -- ask for number of cores (default: 1) --
#BSUB -n 
### -- Choose cpu model
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
### request RAM system-memory
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
### -- set the email address --
### please uncomment the following line and put in your e-mail address,
### if you want to receive e-mail notifications on a non-default address
###BSUB -u s210500@student.dtu.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_out/training%J.out
#BSUB -e job_out/training%J.err
# -- end of LSF options --

# Load env variables
source ~/dev.env

# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
    mkdir ${REPO}/job_out
fi

# Load modules
module load python3/3.10.7
# Activate virtual environment
# If you haven't created it yet, run the following command:
# python3 -m venv .venv
# python3 -m pip install -r requirements.txt
source ${REPO}/.venv/bin/activate

if [[ $? -ne 0 ]]; then
    exit 1
fi

# Run the script
python run.py --train-data <path_to_training.mat> \
       --train-labels <path_to_training_labels.mat> \
       --model-type <model_type>

if [[ $? -ne 0 ]]; then
    exit 1
fi
