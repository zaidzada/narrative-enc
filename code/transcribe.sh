#!/usr/bin/env bash
#SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=32G                 # memory per cpu-core (4G is default)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=encoding      # create a short name for your job
#SBATCH --gres=gpu:1             # get a gpu
#SBATCH -o 'logs/%A-transcribe.log'

source /usr/share/Modules/init/bash
module load anacondapy/2023.07-cuda
conda activate fconv2

echo "${CONDA_PROMPT_MODIFIER}Encoding"
echo "${CONDA_PROMPT_MODIFIER}Requester: $USER"
echo "${CONDA_PROMPT_MODIFIER}Node: $HOSTNAME"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "${CONDA_PROMPT_MODIFIER}SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "${CONDA_PROMPT_MODIFIER}python: $(which python)"
echo "${CONDA_PROMPT_MODIFIER}Start time:" `date`

set -e

whisperx data/stimuli/black_audio.wav --model large-v2 --output_dir transcripts
whisperx data/stimuli/forgot_audio.wav --model large-v2 --output_dir transcripts

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`