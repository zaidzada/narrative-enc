#!/usr/bin/env bash
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o 'logs/%A.log'

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

export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

python "$@"

echo "${CONDA_PROMPT_MODIFIER}End time:" `date`