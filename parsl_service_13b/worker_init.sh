source ~/.bashrc
conda activate ~/environments/anl_llma-13b
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
export ENABLE_SDP_FUSION=1
source /soft/compilers/oneapi/2023.05.15.001/oneapi/compiler/latest/env/vars.sh
source /soft/compilers/oneapi/2023.05.15.001/oneapi/mkl/latest/env/vars.sh
