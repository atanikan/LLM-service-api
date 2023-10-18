#!/bin/bash

#PBS -A Aurora_deployment
#PBS -q workq
#PBS -l select=1
#PBS -l walltime=120:00


source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate

conda activate /lus/gila/projects/Aurora_deployment/atanikanti/environments/openai-ppi-llm-env

cd $PBS_O_WORKDIR

export OPENAI_API_KEY='<REPLACE>'

bash run_prompt_chat_GPT.sh RAD51

#mpirun -np 4 python -u run_generation_with_deepspeed.py -m /home/jmitche1/huggingface/llama2 --ipex --benchmark --input-tokens=1024 --max-new-tokens=128 |& tee llma-70-opt-ipex.log
