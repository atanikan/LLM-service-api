source ~/.bashrc
export HF_HOME=/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=1
export COL_MAJOR=0
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_OP_SYNC=1
export CCL_PROCESS_LAUNCHER=pmix
export FI_PROVIDER=cxi
export PALS_PMI=pmix
export CCL_ATL_TRANSPORT=mpi
export FI_MR_CACHE_MONITOR=disabled
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=32768
export I_MPI_ROOT=/opt/cray/pe/pals/1.2.12/bin/mpiexec
module use -a /home/ftartagl/modulefiles
module load oneapi-testing/2023.2.003.PUBLIC_IDP49422oneapi-testing/2023.2.003.PUBLIC_IDP49422
source /home/jmitche1/70Bccl/libraries.performance.communication.oneccl/build/_install/env/vars.sh
conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b
