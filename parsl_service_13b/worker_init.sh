source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate
conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
export ENABLE_SDP_FUSION=1
source /soft/compilers/oneapi/2023.05.15.001/oneapi/compiler/latest/env/vars.sh
source /soft/compilers/oneapi/2023.05.15.001/oneapi/mkl/latest/env/vars.sh
