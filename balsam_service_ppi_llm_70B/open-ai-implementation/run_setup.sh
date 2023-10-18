#!/bin/bash
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate

#source /home/atanikanti/anaconda3/bin/activate
conda create --prefix /lus/gila/projects/Aurora_deployment/atanikanti/environments/openai-ppi-llm-env python=3.9 --y
conda activate /lus/gila/projects/Aurora_deployment/atanikanti/environments/openai-ppi-llm-env

pip install openai mpi4py pandas
